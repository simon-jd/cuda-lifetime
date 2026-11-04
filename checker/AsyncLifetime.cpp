#include "checker/AsyncLifetime.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <tuple>
using namespace clang;
using namespace ento;
namespace {
using Stream = uint64_t;
enum Memory { Unknown, Pinned, Pageable, Device, Ordered };
enum Access { HostRead, HostWrite, DeviceUse };
// The rule a report belongs to.
enum class Rule : unsigned { CL1 = 1, CL2 = 2, CL3 = 3, CL4 = 4 };
// cudaMemcpyKind.
enum CopyKind : int { HostToHost = 0, HostToDevice = 1, DeviceToHost = 2, DeviceToDevice = 3, DefaultCopy = 4 };
// How a stream is ordered with the legacy default stream (stored in StreamKinds).
enum class StreamKind : unsigned { Unknown = 0, Legacy = 1, Blocking = 2, NonBlocking = 3, PerThread = 4 };
// cudaStreamNonBlocking and cudaEventDisableTiming.
constexpr uint64_t StreamNonBlockingFlag = 0x1, EventDisableTimingFlag = 0x2;
// identity() maps a concrete handle h to 2h + 2 (symbolic handles are odd), so
// the null stream, cudaStreamLegacy ((cudaStream_t)1) and cudaStreamPerThread
// ((cudaStream_t)2) have fixed identities.
constexpr Stream handleIdentity(uint64_t handle) { return handle * 2 + 2; }
constexpr Stream NullStream = handleIdentity(0), LegacyStream = handleIdentity(1), PerThreadStream = handleIdentity(2);
// The effect a model gives a call: cudalifetime::Model::effect, which is also
// the "effect" field of a JSON model (Models.cpp lists the accepted names).
enum class Effect {
  None,
  Other,
  Copy,
  EnqueueCopy,
  Memset,
  MemsetAsync,
  Launch,
  CaptureBegin,
  CaptureEnd,
  GraphLaunch,
  SyncStream,
  SyncDevice,
  SyncEvent,
  EventElapsed,
  EventRecord,
  StreamWaitEvent,
  FreeAsync,
  FreeDevice,
  FreeHost,
  HostRegister,
  HostUnregister,
  MallocHost,
  MallocDevice,
  MallocAsync,
  StreamCreate,
  EventCreate,
  AddressOnly,
  Observe,
  Destroy
};
Effect effectOf(const std::string &name) {
  static const std::map<std::string, Effect> effects = {{"copy", Effect::Copy},
                                                        {"enqueue_copy", Effect::EnqueueCopy},
                                                        {"memset", Effect::Memset},
                                                        {"memset_async", Effect::MemsetAsync},
                                                        {"launch", Effect::Launch},
                                                        {"capture_begin", Effect::CaptureBegin},
                                                        {"capture_end", Effect::CaptureEnd},
                                                        {"graph_launch", Effect::GraphLaunch},
                                                        {"sync_stream", Effect::SyncStream},
                                                        {"sync_device", Effect::SyncDevice},
                                                        {"sync_event", Effect::SyncEvent},
                                                        {"event_elapsed", Effect::EventElapsed},
                                                        {"event_record", Effect::EventRecord},
                                                        {"stream_wait_event", Effect::StreamWaitEvent},
                                                        {"free_async", Effect::FreeAsync},
                                                        {"free_device", Effect::FreeDevice},
                                                        {"free_host", Effect::FreeHost},
                                                        {"host_register", Effect::HostRegister},
                                                        {"host_unregister", Effect::HostUnregister},
                                                        {"malloc_host", Effect::MallocHost},
                                                        {"malloc_device", Effect::MallocDevice},
                                                        {"malloc_async", Effect::MallocAsync},
                                                        {"stream_create", Effect::StreamCreate},
                                                        {"event_create", Effect::EventCreate},
                                                        {"address_only", Effect::AddressOnly},
                                                        {"observe", Effect::Observe},
                                                        {"destroy", Effect::Destroy}};
  if (name.empty())
    return Effect::None;
  auto it = effects.find(name);
  return it == effects.end() ? Effect::Other : it->second;
}
struct Pair {
  uint64_t first = 0, second = 0;
  bool operator==(const Pair &o) const { return first == o.first && second == o.second; }
  bool operator<(const Pair &o) const { return std::tie(first, second) < std::tie(o.first, o.second); }
  void Profile(llvm::FoldingSetNodeID &id) const {
    id.AddInteger(first);
    id.AddInteger(second);
  }
};
struct Pending {
  const MemRegion *region = nullptr;
  Stream stream = 0;
  unsigned seq = 0;
  Access access = HostRead;
  Memory memory = Unknown;
  const Stmt *origin = nullptr;
  const StackFrame *context = nullptr;
  uint64_t bits = 0;
  bool operator==(const Pending &o) const {
    return std::tie(region, stream, seq, access, memory, origin, context, bits) ==
           std::tie(o.region, o.stream, o.seq, o.access, o.memory, o.origin, o.context, o.bits);
  }
  bool operator<(const Pending &o) const {
    return std::tie(region, stream, seq, access, memory, origin, context, bits) <
           std::tie(o.region, o.stream, o.seq, o.access, o.memory, o.origin, o.context, o.bits);
  }
  void Profile(llvm::FoldingSetNodeID &id) const {
    id.AddPointer(region);
    id.AddInteger(stream);
    id.AddInteger(seq);
    id.AddInteger(access);
    id.AddInteger(memory);
    id.AddPointer(origin);
    id.AddPointer(context);
    id.AddInteger(bits);
  }
};
struct Allocation {
  Stream stream = 0;
  unsigned seq = 0;
  bool freed = false;
  Stream freeStream = 0;
  unsigned freeSeq = 0;
  const Stmt *origin = nullptr;
  const StackFrame *context = nullptr;
  bool operator==(const Allocation &o) const {
    return std::tie(stream, seq, freed, freeStream, freeSeq, origin, context) ==
           std::tie(o.stream, o.seq, o.freed, o.freeStream, o.freeSeq, o.origin, o.context);
  }
  void Profile(llvm::FoldingSetNodeID &id) const {
    id.AddInteger(stream);
    id.AddInteger(seq);
    id.AddBoolean(freed);
    id.AddInteger(freeStream);
    id.AddInteger(freeSeq);
    id.AddPointer(origin);
    id.AddPointer(context);
  }
};
} // namespace
REGISTER_SET_WITH_PROGRAMSTATE(PendingOps, Pending)
REGISTER_SET_WITH_PROGRAMSTATE(PreservedVariables, const MemRegion *)
// A host container's buffer, or a view of it, to the container object that owns it.
REGISTER_MAP_WITH_PROGRAMSTATE(ContainerOwners, const MemRegion *, const MemRegion *)
// A view returned by value (a conversion operator) whose object CSA does not
// expose, by the call expression that returned it, to its owner.
REGISTER_MAP_WITH_PROGRAMSTATE(ViewResults, const Expr *, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(Memories, const MemRegion *, unsigned)
REGISTER_MAP_WITH_PROGRAMSTATE(RegisteredExtents, const MemRegion *, uint64_t)
REGISTER_MAP_WITH_PROGRAMSTATE(Sequences, Stream, unsigned)
REGISTER_MAP_WITH_PROGRAMSTATE(StreamKinds, Stream, unsigned)
REGISTER_MAP_WITH_PROGRAMSTATE(StreamValues, Stream, SVal)
REGISTER_MAP_WITH_PROGRAMSTATE(EventValues, uint64_t, SVal)
REGISTER_MAP_WITH_PROGRAMSTATE(WaitBounds, Pair, unsigned)
REGISTER_MAP_WITH_PROGRAMSTATE(HostCompleted, Stream, unsigned)
REGISTER_MAP_WITH_PROGRAMSTATE(EventBounds, Pair, unsigned)
REGISTER_MAP_WITH_PROGRAMSTATE(RecordedEvents, uint64_t, bool)
REGISTER_MAP_WITH_PROGRAMSTATE(EventTimingDisabled, uint64_t, bool)
REGISTER_MAP_WITH_PROGRAMSTATE(Allocations, const MemRegion *, Allocation)
// Pending operations whose buffers reach the arguments of the call being
// evaluated, by whether the callee may write them. Set at PreCall, reported at
// PostCall if CSA did not inline the call.
REGISTER_SET_WITH_PROGRAMSTATE(MutableReach, Pending)
REGISTER_SET_WITH_PROGRAMSTATE(ReadOnlyReach, Pending)
// Streams in stream capture, mapped to the stream that began the capture, and
// events recorded during a capture.
REGISTER_MAP_WITH_PROGRAMSTATE(CaptureOrigins, Stream, Stream)
REGISTER_MAP_WITH_PROGRAMSTATE(CapturedEvents, uint64_t, Stream)

namespace {
class PendingVisitor : public BugReporterVisitor {
  Pending op;

public:
  explicit PendingVisitor(Pending p) : op(p) {}
  void Profile(llvm::FoldingSetNodeID &id) const override {
    static int tag;
    id.AddPointer(&tag);
    op.Profile(id);
  }
  PathDiagnosticPieceRef VisitNode(const ExplodedNode *node, BugReporterContext &context,
                                   PathSensitiveBugReport &) override {
    if (!op.origin || !op.context || node->pred_empty())
      return nullptr;
    if (node->getState()->get<PendingOps>().contains(op) &&
        !(*node->pred_begin())->getState()->get<PendingOps>().contains(op))
      return std::make_shared<PathDiagnosticEventPiece>(
          PathDiagnosticLocation::createBegin(op.origin, context.getSourceManager(), op.context),
          "Asynchronous operation is queued here and remains pending");
    return nullptr;
  }
};
class AllocationVisitor : public BugReporterVisitor {
  const MemRegion *region;
  Allocation allocation;
  bool stillPending;
  bool matches(ProgramStateRef state) const {
    const auto *current = state->get<Allocations>(region);
    return current && current->stream == allocation.stream && current->seq == allocation.seq &&
           current->origin == allocation.origin && current->context == allocation.context;
  }

public:
  AllocationVisitor(const MemRegion *region, Allocation allocation, bool stillPending)
      : region(region), allocation(allocation), stillPending(stillPending) {}
  void Profile(llvm::FoldingSetNodeID &id) const override {
    static int tag;
    id.AddPointer(&tag);
    id.AddPointer(region);
    allocation.Profile(id);
    id.AddBoolean(stillPending);
  }
  PathDiagnosticPieceRef VisitNode(const ExplodedNode *node, BugReporterContext &context,
                                   PathSensitiveBugReport &) override {
    if (!allocation.origin || !allocation.context || node->pred_empty())
      return nullptr;
    // Emit only at the real ExplodedGraph transition that introduces this
    // allocation, including a callee's frame when allocation was inlined.
    if (matches(node->getState()) && !matches((*node->pred_begin())->getState()))
      return std::make_shared<PathDiagnosticEventPiece>(
          PathDiagnosticLocation::createBegin(allocation.origin, context.getSourceManager(), allocation.context),
          stillPending ? "Asynchronous operation is queued here and remains pending"
                       : "Stream-ordered allocation is established here");
    return nullptr;
  }
};
class AsyncLifetimeChecker
    : public Checker<check::PreCall, check::PostCall, eval::Call, check::Location, check::Bind,
                     check::PostStmt<CUDAKernelCallExpr>, check::EndFunction, check::PointerEscape, check::EndAnalysis,
                     check::LiveSymbols, check::BeginFunction, check::BlockEntrance> {
  // One bug type per (rule, level, memory kind, semantic ID), so reports that
  // differ only in those stay distinct, as CSA deduplicates by bug type.
  mutable std::map<std::string, std::unique_ptr<BugType>> bugTypes;
  static const MemRegion *base(SVal v) {
    auto *r = v.getAsRegion();
    return r ? r->StripCasts() : nullptr;
  }
  static bool sameAddress(ProgramStateRef s, const MemRegion *a, const MemRegion *b) {
    if (!a || !b)
      return false;
    if (a == b)
      return true;
    auto equal = s->getStateManager()
                     .getSValBuilder()
                     .evalEQ(s, loc::MemRegionVal(a), loc::MemRegionVal(b))
                     .getAs<DefinedOrUnknownSVal>();
    return equal && !s->assume(*equal, false);
  }
  static const MemRegion *allocationRegion(ProgramStateRef s, const MemRegion *r) {
    if (!r)
      return nullptr;
    auto *root = r->getBaseRegion();
    if (s->get<Allocations>(root))
      return root;
    for (auto entry : s->get<Allocations>())
      if (sameAddress(s, root, entry.first))
        return entry.first;
    return nullptr;
  }
  static uint64_t identity(SVal v) {
    if (auto n = v.getAs<loc::ConcreteInt>())
      return handleIdentity(n->getValue()->getZExtValue());
    if (auto n = v.getAs<nonloc::ConcreteInt>())
      return handleIdentity(n->getValue()->getZExtValue());
    if (auto *r = base(v))
      return reinterpret_cast<uintptr_t>(r) | 1;
    if (auto s = v.getAsSymbol())
      return reinterpret_cast<uintptr_t>(s) | 1;
    return 0;
  }
  static bool sameValue(ProgramStateRef s, SVal a, SVal b) {
    // SValBuilder's pointer/integer comparison requires a cast first. Unknown
    // CUDA operands must not reach evalEQ as incompatible Loc/NonLoc values.
    if (a.isUnknownOrUndef() || b.isUnknownOrUndef() || bool(a.getAs<Loc>()) != bool(b.getAs<Loc>()))
      return false;
    auto equal = s->getStateManager().getSValBuilder().evalEQ(s, a, b).getAs<DefinedOrUnknownSVal>();
    return equal && !s->assume(*equal, false);
  }
  static uint64_t streamIdentity(ProgramStateRef s, SVal value) {
    uint64_t raw = identity(value);
    auto &builder = s->getStateManager().getSValBuilder();
    if (raw == NullStream || sameValue(s, value, builder.makeZeroVal(builder.getContext().VoidPtrTy)))
      return defaultStream();
    for (auto entry : s->get<StreamValues>())
      if (sameValue(s, entry.second, value))
        return entry.first;
    return identity(value);
  }
  static uint64_t eventIdentity(ProgramStateRef s, SVal value) {
    for (auto entry : s->get<EventValues>())
      if (sameValue(s, entry.second, value))
        return entry.first;
    return identity(value);
  }
  static unsigned sequence(ProgramStateRef s, Stream stream) {
    auto *n = s->get<Sequences>(stream);
    return n ? *n : 0;
  }
  static unsigned completed(ProgramStateRef s, Stream stream) {
    auto *n = s->get<HostCompleted>(stream);
    return n ? *n : 0;
  }
  static Stream defaultStream() { return cudalifetime::info().perThread ? PerThreadStream : LegacyStream; }
  static bool capturing(ProgramStateRef s, Stream stream) { return stream && s->get<CaptureOrigins>(stream); }
  static void unknownDirection() {
    cudalifetime::info().coverageUnknown.insert("cudaMemcpyDefault copy with an operand of unknown memory kind: the "
                                                "copy direction for that operand is not modeled");
  }
  static void noteCapture() {
    cudalifetime::info().coverageUnknown.insert("work enqueued during stream capture is recorded into a graph, not "
                                                "executed; graph contents and launches are not modeled");
  }
  static Memory memory(ProgramStateRef s, const MemRegion *r, uint64_t bits = 0) {
    if (!r)
      return Unknown;
    for (auto entry : s->get<Memories>()) {
      if (const auto *extent = s->get<RegisteredExtents>(entry.first)) {
        auto ro = r->getAsOffset(), eo = entry.first->getAsOffset();
        if (!ro.isValid() || !eo.isValid() || ro.hasSymbolicOffset() || eo.hasSymbolicOffset() ||
            !sameAddress(s, ro.getRegion(), eo.getRegion()) || !bits || !*extent)
          continue;
        __int128 start = ro.getOffset(), registeredStart = eo.getOffset();
        if (start >= registeredStart && start + bits <= registeredStart + *extent)
          return Pinned;
      } else if (r == entry.first || r->isSubRegionOf(entry.first) || sameAddress(s, r, entry.first))
        return static_cast<Memory>(entry.second);
    }
    if (r->hasMemorySpace<StackSpaceRegion>(s))
      return Pageable;
    return Unknown;
  }
  static StreamKind kind(ProgramStateRef s, Stream stream) {
    if (stream == NullStream)
      return cudalifetime::info().perThread ? StreamKind::PerThread : StreamKind::Legacy;
    if (stream == LegacyStream)
      return StreamKind::Legacy;
    if (stream == PerThreadStream)
      return StreamKind::PerThread;
    auto *v = s->get<StreamKinds>(stream);
    return v ? static_cast<StreamKind>(*v) : StreamKind::Unknown;
  }
  static bool distinct(ProgramStateRef s, Stream a, Stream b) {
    if (!a || !b || a == b)
      return false;
    const bool aCreated = s->get<StreamKinds>(a), bCreated = s->get<StreamKinds>(b);
    const bool aReserved = a == LegacyStream || a == PerThreadStream,
               bReserved = b == LegacyStream || b == PerThreadStream;
    if ((aCreated && (bCreated || bReserved)) || (bCreated && aReserved) || (aReserved && bReserved))
      return true;
    const auto *av = s->get<StreamValues>(a), *bv = s->get<StreamValues>(b);
    if (!av || !bv)
      return false;
    if (av->isUnknownOrUndef() || bv->isUnknownOrUndef() || bool(av->getAs<Loc>()) != bool(bv->getAs<Loc>()))
      return false;
    auto equal = s->getStateManager().getSValBuilder().evalEQ(s, *av, *bv).getAs<DefinedOrUnknownSVal>();
    return equal && !s->assume(*equal, true);
  }
  static bool ordered(ProgramStateRef s, Stream waiter, Stream source, unsigned seq) {
    if (completed(s, source) >= seq)
      return true;
    if (!waiter || !source)
      return false;
    if (waiter == source)
      return true;
    const auto *wv = s->get<StreamValues>(waiter), *sv = s->get<StreamValues>(source);
    if (wv && sv && sameValue(s, *wv, *sv))
      return true;
    auto *bound = s->get<WaitBounds>({waiter, source});
    return bound && *bound >= seq;
  }
  static ProgramStateRef bound(ProgramStateRef s, Stream waiter, Stream source, unsigned seq) {
    auto *old = s->get<WaitBounds>({waiter, source});
    if (!old || *old < seq)
      s = s->set<WaitBounds>({waiter, source}, seq);
    return s;
  }
  static ProgramStateRef enqueue(ProgramStateRef s, Stream stream, SVal value = UnknownVal()) {
    if (!value.isUnknownOrUndef())
      s = s->set<StreamValues>(stream, value);
    // Legacy's device ordering is independent of host completion.
    const StreamKind k = kind(s, stream);
    for (auto it : s->get<Sequences>()) {
      const StreamKind other = kind(s, it.first);
      // CUDA's per-thread default stream is blocking with respect to an
      // explicitly used legacy stream, but does not order ordinary streams.
      // CUDA 13.2.2 archive: stream-sync-behavior#per-thread-default-stream.
      if ((k == StreamKind::Legacy && (other == StreamKind::Blocking || other == StreamKind::PerThread)) ||
          ((k == StreamKind::Blocking || k == StreamKind::PerThread) && other == StreamKind::Legacy)) {
        s = bound(s, stream, it.first, it.second);
        for (auto wait : s->get<WaitBounds>())
          if (wait.first.first == it.first)
            s = bound(s, stream, wait.first.second, wait.second);
      }
    }
    return s->set<Sequences>(stream, sequence(s, stream) + 1);
  }
  static ProgramStateRef retire(ProgramStateRef s, Stream stream, unsigned seq) {
    if (seq > completed(s, stream))
      s = s->set<HostCompleted>(stream, seq);
    for (auto op : s->get<PendingOps>())
      if (op.stream == stream && op.seq <= seq)
        s = s->remove<PendingOps>(op);
    return s;
  }
  static ProgramStateRef syncStream(ProgramStateRef s, Stream stream) {
    if (!stream)
      return s;
    s = retire(s, stream, sequence(s, stream));
    for (auto wait : s->get<WaitBounds>())
      if (wait.first.first == stream)
        s = retire(s, wait.first.second, wait.second);
    return s;
  }
  static bool overlaps(ProgramStateRef s, const MemRegion *a, const MemRegion *b) {
    return a && b && (a == b || a->isSubRegionOf(b) || b->isSubRegionOf(a) || sameAddress(s, a, b));
  }
  static unsigned rangeOverlap(ProgramStateRef s, const MemRegion *a, uint64_t aBits, const MemRegion *b,
                               uint64_t bBits) {
    auto ao = a->getAsOffset(), bo = b->getAsOffset();
    if (ao.isValid() && bo.isValid() && sameAddress(s, ao.getRegion(), bo.getRegion())) {
      if (ao.hasSymbolicOffset() || bo.hasSymbolicOffset() || !aBits || !bBits)
        return 1;
      __int128 aStart = ao.getOffset(), bStart = bo.getOffset();
      return aStart < bStart + bBits && bStart < aStart + aBits ? 2 : 0;
    }
    return overlaps(s, a, b) ? 1 : 0;
  }
  void report(CheckerContext &c, ProgramStateRef s, Rule rule, const std::string &description, Memory mem,
              const char *semantic, const Pending *op = nullptr, bool contract = false, bool informational = false,
              const Allocation *allocation = nullptr, const MemRegion *allocationKey = nullptr) const {
    auto *node = c.generateNonFatalErrorNode(s);
    if (!node)
      return;
    const char *names[] = {"unknown", "pinned", "pageable", "device", "stream-ordered"};
    const bool certain = !contract && (mem == Pinned || mem == Ordered);
    const cudalifetime::ReportKind kind{"CL" + std::to_string(static_cast<unsigned>(rule)),
                                        informational ? "note"
                                        : certain     ? "error"
                                                      : "warning",
                                        names[mem], semantic};
    const std::string key = kind.rule + "/" + kind.level + "/" + kind.memory + "/" + kind.semantic;
    auto &type = bugTypes[key];
    if (!type) {
      type = std::make_unique<BugType>(this, kind.rule, key);
      cudalifetime::info().reports[key] = kind;
    }
    auto r = std::make_unique<PathSensitiveBugReport>(*type, description, node);
    if (op) {
      r->markInteresting(op->region);
      r->addVisitor(std::make_unique<PendingVisitor>(*op));
      if (op->origin && op->context)
        r->addNote("Asynchronous operation is queued here; completion has not been established",
                   PathDiagnosticLocation::createBegin(op->origin, c.getSourceManager(), op->context));
    }
    if (allocation && allocationKey) {
      r->markInteresting(allocationKey);
      r->addVisitor(std::make_unique<AllocationVisitor>(allocationKey, *allocation,
                                                        completed(s, allocation->stream) < allocation->seq));
    }
    c.emitReport(std::move(r));
  }
  static SVal kernelValue(const Expr *expr, CheckerContext &c) {
    SVal value = c.getSVal(expr);
    if (!value.isUnknownOrUndef())
      return value;
    QualType originalType = expr->getType();
    expr = expr->IgnoreParenCasts();
    if (const auto *arg = dyn_cast<CXXDefaultArgExpr>(expr))
      value = kernelValue(arg->getExpr(), c);
    else if (const auto *ref = dyn_cast<DeclRefExpr>(expr)) {
      if (const auto *decl = dyn_cast<VarDecl>(ref->getDecl()))
        value = c.getState()->getSVal(c.getState()->getLValue(decl, c.getStackFrame()));
    } else if (const auto *literal = dyn_cast<IntegerLiteral>(expr))
      value = c.getSValBuilder().makeIntVal(literal->getValue(), literal->getType()->isUnsignedIntegerType());
    else if (isa<CXXNullPtrLiteralExpr>(expr))
      value = c.getSValBuilder().makeZeroVal(originalType);
    // ExprEngine skips CUDA operands. Reapply their source cast after reading
    // the store/literal so (cudaStream_t)1 remains a pointer, not an integer.
    return c.getSValBuilder().evalCast(value, originalType, expr->getType());
  }
  void deviceUse(CheckerContext &c, ProgramStateRef s, const MemRegion *r, Stream stream, const char *action) const {
    if (!r || capturing(s, stream))
      return;
    if (auto *key = allocationRegion(s, r)) {
      const auto *alloc = s->get<Allocations>(key);
      if (alloc->freed)
        report(c, s, Rule::CL3, std::string(action) + " uses an allocation after its asynchronous free was issued",
               Ordered, "S6", nullptr, false, false, alloc, key);
      else if (!ordered(s, stream, alloc->stream, alloc->seq))
        report(c, s, Rule::CL3,
               std::string(action) + " uses a stream-ordered allocation without an allocation dependency", Ordered,
               "S5", nullptr, !distinct(s, stream, alloc->stream), false, alloc, key);
    }
  }
  static std::string name(const CallEvent &call) { return declName(dyn_cast_or_null<FunctionDecl>(call.getDecl())); }
  static std::string declName(const FunctionDecl *fd) {
    if (fd) {
      auto qualified = fd->getQualifiedNameAsString();
      if (cudalifetime::info().models.count(qualified))
        return qualified;
      auto n = fd->getNameAsString();
      for (const char *suffix : {"_ptsz", "_ptds"})
        if (n.size() > 5 && n.compare(n.size() - 5, 5, suffix) == 0)
          n.resize(n.size() - 5);
      return n;
    }
    return {};
  }
  // Device, version and occupancy queries and settings: no stream work, no user buffer.
  static bool deviceQuery(const std::string &n) {
    static const std::set<std::string> names = {"cudaGetDevice",
                                                "cudaSetDevice",
                                                "cudaGetDeviceCount",
                                                "cudaGetDeviceProperties",
                                                "cudaDeviceGetAttribute",
                                                "cudaDeviceCanAccessPeer",
                                                "cudaDriverGetVersion",
                                                "cudaRuntimeGetVersion",
                                                "cudaMemGetInfo",
                                                "cudaDeviceSetLimit",
                                                "cudaDeviceGetLimit",
                                                "cudaFuncSetAttribute",
                                                "cudaFuncGetAttributes",
                                                "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                                                "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                                                "cudaCreateChannelDesc",
                                                "cudaStreamGetId",
                                                "cudaStreamGetPriority",
                                                "cudaDeviceGetStreamPriorityRange",
                                                "cudaGetDeviceFlags",
                                                "cudaSetDeviceFlags",
                                                "cudaChooseDevice",
                                                "cudaProfilerStart",
                                                "cudaProfilerStop"};
    return names.count(n) != 0;
  }
  static cudalifetime::Model model(const std::string &n) {
    if (auto it = cudalifetime::info().models.find(n); it != cudalifetime::info().models.end())
      return it->second;
    cudalifetime::Model m;
    m.function = n;
    if (n == "cudaMemcpyAsync") {
      m.effect = "enqueue_copy";
      m.args = {{"dst", 0}, {"src", 1}, {"size", 2}, {"kind", 3}, {"stream", 4}};
    } else if (n == "cudaMemcpyBatchAsync") {
      // Arrays of operands; the direction follows from the pointers, as with
      // cudaMemcpyDefault. Only the first entry is modeled.
      m.effect = "enqueue_copy";
      m.args = {{"dst", 0}, {"src", 1}, {"size", 2}, {"count", 3}, {"stream", 7}};
      m.constants = {{"kind", 4}};
      m.elements = {"dst", "src", "size"};
    } else if (n == "cudaMemcpy2DAsync") {
      m.effect = "enqueue_copy";
      m.args = {{"dst", 0}, {"src", 2}, {"kind", 6}, {"stream", 7}};
    } else if (n == "cudaMemcpy") {
      m.effect = "copy";
      m.args = {{"dst", 0}, {"src", 1}, {"size", 2}, {"kind", 3}};
    } else if (n == "cudaMemcpy2D") {
      m.effect = "copy";
      m.args = {{"dst", 0}, {"src", 2}, {"kind", 6}};
    } else if (n == "cudaMemset") {
      m.effect = "memset";
      m.args = {{"ptr", 0}, {"size", 2}};
    } else if (n == "cudaMemsetAsync") {
      m.effect = "memset_async";
      m.args = {{"ptr", 0}, {"size", 2}, {"stream", 3}};
    } else if (n == "cudaStreamBeginCapture" || n == "cudaStreamBeginCaptureToGraph") {
      m.effect = "capture_begin";
      m.args = {{"stream", 0}};
    } else if (n == "cudaStreamEndCapture") {
      m.effect = "capture_end";
      m.args = {{"stream", 0}, {"ptr", 1}};
    } else if (n == "cudaGraphLaunch") {
      m.effect = "graph_launch";
      m.args = {{"stream", 1}};
    } else if (n == "cudaPointerGetAttributes" || n == "cudaHostGetDevicePointer" || n == "cudaHostGetFlags")
      m.effect = "address_only";
    else if (n == "cudaMallocHost" || n == "cudaHostAlloc") {
      m.effect = "malloc_host";
      m.args = {{"ptr", 0}};
    } else if (n == "cudaHostRegister") {
      m.effect = "host_register";
      m.args = {{"ptr", 0}, {"size", 1}};
    } else if (n == "cudaHostUnregister") {
      m.effect = "host_unregister";
      m.args = {{"ptr", 0}};
    } else if (n == "cudaMalloc") {
      m.effect = "malloc_device";
      m.args = {{"ptr", 0}};
    } else if (n == "cudaMallocAsync" || n == "cudaMallocFromPoolAsync") {
      m.effect = "malloc_async";
      m.args = {{"ptr", 0}, {"stream", n == "cudaMallocAsync" ? 2u : 3u}};
    } else if (n == "cudaFreeAsync") {
      m.effect = "free_async";
      m.args = {{"ptr", 0}, {"stream", 1}};
    } else if (n == "cudaFreeHost") {
      m.effect = "free_host";
      m.args = {{"ptr", 0}};
    } else if (n == "cudaFree") {
      m.effect = "free_device";
      m.args = {{"ptr", 0}};
    } else if (n == "cudaDeviceSynchronize")
      m.effect = "sync_device";
    else if (n == "cudaStreamSynchronize" || n == "cudaStreamQuery") {
      m.effect = "sync_stream";
      m.args = {{"stream", 0}};
    } else if (n == "cudaEventSynchronize" || n == "cudaEventQuery") {
      m.effect = "sync_event";
      m.args = {{"event", 0}};
    } else if (n == "cudaEventElapsedTime") {
      m.effect = "event_elapsed";
      m.args = {{"ms", 0}, {"start", 1}, {"end", 2}};
    } else if (n == "cudaEventRecord" || n == "cudaEventRecordWithFlags") {
      m.effect = "event_record";
      m.args = {{"event", 0}, {"stream", 1}};
    } else if (n == "cudaStreamWaitEvent") {
      m.effect = "stream_wait_event";
      m.args = {{"stream", 0}, {"event", 1}};
    } else if (n == "cudaStreamCreate" || n == "cudaStreamCreateWithFlags" || n == "cudaStreamCreateWithPriority") {
      m.effect = "stream_create";
      m.args = {{"ptr", 0}, {"flags", 1}};
    } else if (n == "cudaEventCreate" || n == "cudaEventCreateWithFlags") {
      m.effect = "event_create";
      m.args = {{"ptr", 0}};
      if (n == "cudaEventCreateWithFlags")
        m.args["flags"] = 1;
    } else if (n == "cudaStreamDestroy" || n == "cudaEventDestroy")
      m.effect = "destroy";
    else if (n == "cudaGetLastError" || n == "cudaPeekAtLastError" || n == "cudaGetErrorString" ||
             n == "cudaGetErrorName" || n == "cudaGraphInstantiate" || n == "cudaGraphInstantiateWithFlags" ||
             n == "cudaGraphDestroy" || n == "cudaGraphExecDestroy" || deviceQuery(n))
      m.effect = "observe";
    return m;
  }

public:
  // Summaries of bodies: the copy a function queues on its own parameters,
  // composed from the CUDA calls, models and other bodies it calls. They are
  // flow-insensitive "may" summaries: a copy on any path counts, and a body
  // that synchronizes anything gets none, since it may complete its copies.
  // A wrapper whose only data member is a handle (cuda_stream_view, stream_ref).
  static bool handleRecord(QualType type) {
    if (!type.isNull() && type->isPointerType())
      type = type->getPointeeType();
    const auto *record = type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl();
    if (!record || !(record = record->getDefinition()) || record->getNumBases() || record->isPolymorphic())
      return false;
    return std::distance(record->field_begin(), record->field_end()) == 1;
  }
  struct Forwarded {
    // level -1: the address of the parameter; 0: its value; 1: an element of
    // the array it points to.
    int param = -1, level = 0;
    bool constant = false, receiver = false;
    int64_t value = 0;
    bool valid() const { return param >= 0 || constant || receiver; }
  };
  // A body that, on every path to its normal exit, calls the function a
  // parameter points to (RAFT's interruptible wait polls the query its caller
  // passes, cudaStreamQuery or cudaEventQuery): which parameter, and where the
  // call's arguments come from.
  struct Through {
    unsigned callee = 0;
    std::vector<Forwarded> args;
  };
  struct Summary {
    bool done = false, synchronizes = false, copies = false;
    std::optional<cudalifetime::Model> copy, sync;
    std::optional<Through> through;
  };
  // The parameter of fd that a call's callee expression reads, when the call
  // goes through a function pointer (or reference) parameter.
  static int calleeParameter(const CallExpr *call, const FunctionDecl *fd) {
    const Expr *callee = call->getCallee()->IgnoreParenImpCasts();
    if (const auto *unary = dyn_cast<UnaryOperator>(callee); unary && unary->getOpcode() == UO_Deref)
      callee = unary->getSubExpr()->IgnoreParenImpCasts();
    const auto *ref = dyn_cast<DeclRefExpr>(callee);
    const auto *param = ref ? dyn_cast<ParmVarDecl>(ref->getDecl()) : nullptr;
    if (!param || !param->getType().getNonReferenceType()->isFunctionPointerType())
      return -1;
    for (unsigned i = 0; i < fd->getNumParams(); ++i)
      if (fd->getParamDecl(i) == param)
        return static_cast<int>(i);
    return -1;
  }
  // The function an argument names (cudaStreamQuery, &cudaStreamQuery).
  static const FunctionDecl *referencedFunction(const Expr *e) {
    e = e->IgnoreParenImpCasts();
    if (const auto *unary = dyn_cast<UnaryOperator>(e); unary && unary->getOpcode() == UO_AddrOf)
      e = unary->getSubExpr()->IgnoreParenImpCasts();
    const auto *ref = dyn_cast<DeclRefExpr>(e);
    return ref ? dyn_cast<FunctionDecl>(ref->getDecl()) : nullptr;
  }
  // The model of a call to a body summarized as calling through a parameter,
  // once the function passed for it is known: the target's synchronization,
  // with its operands taken from the arguments the body forwards to it.
  static std::optional<cudalifetime::Model> resolveThrough(const Through &through, const FunctionDecl *target) {
    const cudalifetime::Model m = model(declName(target));
    const Effect e = effectOf(m.effect);
    if (e != Effect::SyncStream && e != Effect::SyncEvent && e != Effect::SyncDevice)
      return std::nullopt;
    cudalifetime::Model out;
    out.function = declName(target);
    out.effect = m.effect;
    out.assumption = true;
    for (const auto &[key, index] : m.args) {
      if (index >= through.args.size())
        return std::nullopt;
      const Forwarded &f = through.args[index];
      if (f.param >= 0 && f.level == 0)
        out.args[key] = static_cast<unsigned>(f.param);
      else if (f.constant)
        out.constants[key] = f.value;
      else
        return std::nullopt;
    }
    return out;
  }
  // Which parameter of fd an argument expression passes on, or its constant
  // value. Buffers forward as a pointer parameter or the address of a
  // reference parameter; a handle may also be read through a wrapper's
  // accessor or conversion (stream.value(), a cuda_stream_view to stream_ref).
  static Forwarded forwarded(const Expr *e, const FunctionDecl *fd, bool handle) {
    Forwarded out;
    for (unsigned depth = 0; e && depth < 16; ++depth) {
      e = e->IgnoreImplicit()->IgnoreParens();
      if (const auto *cast = dyn_cast<ExplicitCastExpr>(e)) {
        e = cast->getSubExpr();
        continue;
      }
      if (const auto *temp = dyn_cast<MaterializeTemporaryExpr>(e)) {
        e = temp->getSubExpr();
        continue;
      }
      if (handle) {
        if (const auto *construct = dyn_cast<CXXConstructExpr>(e); construct && construct->getNumArgs() == 1) {
          e = construct->getArg(0);
          continue;
        }
        if (const auto *member = dyn_cast<CXXMemberCallExpr>(e);
            member && !member->getNumArgs() && member->getImplicitObjectArgument() &&
            handleRecord(member->getImplicitObjectArgument()->getType())) {
          e = member->getImplicitObjectArgument();
          continue;
        }
      }
      break;
    }
    if (!e)
      return out;
    // A handle the receiver holds: this->stream_, or value() on this.
    if (handle) {
      const Expr *object = e;
      if (const auto *field = dyn_cast<MemberExpr>(e))
        object = field->getBase()->IgnoreParenImpCasts();
      if (isa<CXXThisExpr>(object)) {
        out.receiver = handleRecord(object->getType());
        return out;
      }
    }
    auto index = [&](const ValueDecl *decl) -> int {
      const auto *param = dyn_cast_or_null<ParmVarDecl>(decl);
      if (!param)
        return -1;
      for (unsigned i = 0; i < fd->getNumParams(); ++i)
        if (fd->getParamDecl(i) == param)
          return static_cast<int>(i);
      return -1;
    };
    if (const auto *unary = dyn_cast<UnaryOperator>(e); unary && unary->getOpcode() == UO_AddrOf) {
      // &value of a reference parameter is the address the caller passed; &p
      // of any other parameter is the address of the callee's copy of it.
      if (const auto *ref = dyn_cast<DeclRefExpr>(unary->getSubExpr()->IgnoreParens())) {
        out.param = index(ref->getDecl());
        if (!ref->getDecl()->getType()->isReferenceType())
          out.level = -1;
      }
      return out;
    }
    const Expr *array = nullptr;
    if (const auto *subscript = dyn_cast<ArraySubscriptExpr>(e))
      array = subscript->getBase();
    else if (const auto *unary = dyn_cast<UnaryOperator>(e); unary && unary->getOpcode() == UO_Deref)
      array = unary->getSubExpr();
    if (array) {
      if (const auto *ref = dyn_cast<DeclRefExpr>(array->IgnoreParenImpCasts());
          ref && ref->getDecl()->getType()->isPointerType()) {
        out.param = index(ref->getDecl());
        out.level = 1;
      }
      return out;
    }
    if (const auto *ref = dyn_cast<DeclRefExpr>(e)) {
      QualType type = ref->getDecl()->getType().getNonReferenceType();
      if (type->isPointerType() || type->isIntegralOrEnumerationType() || (handle && type->getAsCXXRecordDecl()))
        out.param = index(ref->getDecl());
      if (out.param >= 0)
        return out;
    }
    Expr::EvalResult result;
    if (!e->isValueDependent() && !e->isTypeDependent() && e->EvaluateAsInt(result, fd->getASTContext())) {
      out.constant = true;
      out.value = result.Val.getInt().getExtValue();
    }
    return out;
  }
  static bool mayReturn(const Stmt *statement) {
    struct Returns : RecursiveASTVisitor<Returns> {
      bool found = false;
      bool VisitReturnStmt(ReturnStmt *) {
        found = true;
        return false;
      }
      bool VisitCoreturnStmt(CoreturnStmt *) {
        found = true;
        return false;
      }
      bool TraverseLambdaExpr(LambdaExpr *) { return true; } // returns from the lambda only
    } returns;
    returns.TraverseStmt(const_cast<Stmt *>(statement));
    return returns.found;
  }
  // Declared noreturn, or every path through the body ends in a throw or a
  // noreturn call (a syntactic check: statements in sequence, if/else).
  static bool endsInThrow(const Stmt *statement, unsigned depth = 0) {
    if (!statement || depth > 16)
      return false;
    if (const auto *expr = dyn_cast<Expr>(statement)) {
      expr = expr->IgnoreImplicit()->IgnoreParens();
      if (isa<CXXThrowExpr>(expr))
        return true;
      if (const auto *call = dyn_cast<CallExpr>(expr))
        return call->getDirectCallee() && call->getDirectCallee()->isNoReturn();
      return false;
    }
    if (const auto *block = dyn_cast<CompoundStmt>(statement)) {
      // A statement that may return first (if (ok) return;) leaves the
      // function before a later throw.
      for (const auto *child : block->body()) {
        if (endsInThrow(child, depth + 1))
          return true;
        if (mayReturn(child))
          return false;
      }
      return false;
    }
    if (const auto *branch = dyn_cast<IfStmt>(statement))
      return endsInThrow(branch->getThen(), depth + 1) && endsInThrow(branch->getElse(), depth + 1);
    return false;
  }
  // A block that ends its path: a throw (the CFG keeps it as the block's last
  // statement, with the exit or a handler as successor), or a call that never returns.
  static bool deadEnd(const CFGBlock *block) {
    if (block->hasNoReturnElement() || isa_and_nonnull<CXXThrowExpr>(block->getTerminatorStmt()))
      return true;
    for (const auto &element : *block)
      if (auto statement = element.getAs<CFGStmt>()) {
        if (isa<CXXThrowExpr>(statement->getStmt()))
          return true;
        if (const auto *inner = dyn_cast<CallExpr>(statement->getStmt());
            inner && inner->getDirectCallee() && neverReturns(inner->getDirectCallee()))
          return true;
      }
    return false;
  }
  // The blocks of a CFG from which every path reaches a dead end before the
  // function's exit (an error branch that formats a message and throws).
  static llvm::BitVector doomedBlocks(const CFG &cfg) {
    llvm::BitVector doomed(cfg.getNumBlockIDs());
    for (const CFGBlock *block : cfg)
      if (block && block != &cfg.getExit() && deadEnd(block))
        doomed.set(block->getBlockID());
    for (bool changed = true; changed;) {
      changed = false;
      for (const CFGBlock *block : cfg) {
        if (!block || block == &cfg.getExit() || doomed.test(block->getBlockID()))
          continue;
        bool any = false, all = true;
        for (const auto &next : block->succs())
          if (const CFGBlock *successor = next.getReachableBlock()) {
            any = true;
            all &= doomed.test(successor->getBlockID());
          }
        if (any && all) {
          doomed.set(block->getBlockID());
          changed = true;
        }
      }
    }
    return doomed;
  }
  // Whether every path from the body's entry to its normal exit passes the
  // call: paths that throw or call a function that never returns don't count.
  static bool alwaysReached(const FunctionDecl *fd, const CallExpr *call) {
    auto cfg = CFG::buildCFG(fd, fd->getBody(), &fd->getASTContext(), CFG::BuildOptions());
    if (!cfg)
      return false;
    llvm::SmallPtrSet<const CFGBlock *, 4> at;
    for (const CFGBlock *block : *cfg)
      for (const auto &element : *block)
        if (auto statement = element.getAs<CFGStmt>(); statement && statement->getStmt() == call)
          at.insert(block);
    if (at.empty())
      return false;
    std::vector<const CFGBlock *> work{&cfg->getEntry()};
    llvm::SmallPtrSet<const CFGBlock *, 32> seen;
    while (!work.empty()) {
      const CFGBlock *block = work.back();
      work.pop_back();
      if (!block || !seen.insert(block).second || at.count(block))
        continue;
      if (block == &cfg->getExit())
        return false;
      if (deadEnd(block))
        continue;
      for (const auto &next : block->succs())
        work.push_back(next.getReachableBlock());
    }
    return true;
  }
  static bool neverReturns(const FunctionDecl *fd) {
    if (fd->isNoReturn())
      return true;
    const FunctionDecl *definition = fd->getDefinition();
    return definition && definition->hasBody() && endsInThrow(definition->getBody());
  }
  static const Summary &summaryOf(const FunctionDecl *fd) {
    static std::map<const FunctionDecl *, Summary> memo;
    fd = fd->getCanonicalDecl();
    auto &entry = memo[fd];
    if (entry.done)
      return entry;
    entry.done = true; // a recursive call sees an empty summary
    const FunctionDecl *definition = fd->getDefinition();
    if (!definition || !definition->hasBody() || definition->isDependentContext())
      return entry;
    struct Calls : RecursiveASTVisitor<Calls> {
      std::vector<const CallExpr *> calls;
      bool VisitCallExpr(CallExpr *call) {
        calls.push_back(call);
        return true;
      }
      bool TraverseLambdaExpr(LambdaExpr *) { return true; } // runs only if called
    } visitor;
    visitor.TraverseStmt(definition->getBody());
    Summary result;
    result.done = true;
    std::vector<cudalifetime::Model> copies;
    auto forwardsAll = [&](const CallExpr *call) {
      std::vector<Forwarded> args;
      for (const Expr *argument : call->arguments())
        args.push_back(forwarded(argument, definition, true));
      return args;
    };
    for (const CallExpr *call : visitor.calls) {
      const FunctionDecl *callee = call->getDirectCallee();
      if (!callee) {
        if (int param = calleeParameter(call, definition);
            param >= 0 && !result.through && !neverReturns(definition) && alwaysReached(definition, call))
          result.through = Through{static_cast<unsigned>(param), forwardsAll(call)};
        continue;
      }
      // A callee that never returns (an error handler that synchronizes, then
      // throws) ends its path: what it does is never followed by the caller's code.
      if (isa<CXXOperatorCallExpr>(call) || neverReturns(callee))
        continue;
      cudalifetime::Model m = model(declName(callee));
      if (m.effect.empty() && !cudalifetime::isCudaApi(callee)) {
        const Summary &inner = summaryOf(callee);
        result.synchronizes |= inner.synchronizes;
        if (inner.copy)
          m = *inner.copy;
        else if (inner.sync)
          m = *inner.sync;
        else if (inner.through && inner.through->callee < call->getNumArgs()) {
          const Expr *passed = call->getArg(inner.through->callee);
          if (const FunctionDecl *target = referencedFunction(passed)) {
            // The function is named here: the call is that function's wait.
            if (auto resolved = resolveThrough(*inner.through, target)) {
              m = *resolved;
              result.synchronizes = true;
            }
          } else if (Forwarded f = forwarded(passed, definition, false);
                     f.param >= 0 && f.level == 0 && !result.through && !neverReturns(definition) &&
                     alwaysReached(definition, call)) {
            // This body passes its own function parameter on.
            Through through{static_cast<unsigned>(f.param), {}};
            for (const Forwarded &arg : inner.through->args)
              through.args.push_back(arg.param >= 0 && arg.level == 0 && unsigned(arg.param) < call->getNumArgs()
                                         ? forwarded(call->getArg(arg.param), definition, true)
                                     : arg.constant ? arg
                                                    : Forwarded());
            result.through = through;
          }
        }
      }
      const Effect e = effectOf(m.effect);
      if (e == Effect::SyncStream || e == Effect::SyncDevice || e == Effect::SyncEvent || e == Effect::EventElapsed ||
          e == Effect::Copy)
        result.synchronizes = true;
      // A synchronization on every path through the body (a "must" claim,
      // unlike copies) makes a summary of its own.
      if ((e == Effect::SyncStream || e == Effect::SyncDevice || e == Effect::SyncEvent) && !result.sync &&
          !neverReturns(definition) && alwaysReached(definition, call)) {
        cudalifetime::Model sync;
        sync.function = definition->getQualifiedNameAsString();
        sync.effect = m.effect;
        sync.assumption = true;
        Forwarded f;
        if (e != Effect::SyncDevice) {
          const std::string key = e == Effect::SyncEvent ? "event" : "stream";
          if (auto constant = m.constants.find(key); constant != m.constants.end()) {
            f.constant = true;
            f.value = constant->second;
          } else if (m.receiver.count(key)) {
            if (const auto *member = dyn_cast<CXXMemberCallExpr>(call); member && member->getImplicitObjectArgument())
              f = forwarded(member->getImplicitObjectArgument(), definition, true);
          } else if (auto it = m.args.find(key); it != m.args.end() && it->second < call->getNumArgs())
            f = forwarded(call->getArg(it->second), definition, true);
          if (f.param >= 0 && f.level == 0)
            sync.args[key] = static_cast<unsigned>(f.param);
          else if (f.receiver)
            sync.receiver.insert(key);
          else if (f.constant && e == Effect::SyncStream)
            sync.constants[key] = f.value;
          else
            f = Forwarded();
        }
        if (e == Effect::SyncDevice || f.valid())
          result.sync = sync;
      }
      if (e != Effect::EnqueueCopy)
        continue;
      cudalifetime::Model copy;
      copy.function = definition->getQualifiedNameAsString();
      copy.effect = m.effect;
      copy.assumption = true;
      bool forwardsBuffer = false, forwardsStream = false;
      for (const char *key : {"dst", "src", "size", "kind", "stream", "count"}) {
        Forwarded f;
        if (auto constant = m.constants.find(key); constant != m.constants.end()) {
          f.constant = true;
          f.value = constant->second;
        } else if (m.receiver.count(key)) {
          if (const auto *member = dyn_cast<CXXMemberCallExpr>(call); member && member->getImplicitObjectArgument())
            f = forwarded(member->getImplicitObjectArgument(), definition, true);
        } else if (auto it = m.args.find(key); it != m.args.end() && it->second < call->getNumArgs()) {
          f = forwarded(call->getArg(it->second), definition, std::string(key) == "stream");
          if (m.elements.count(key)) {
            // The callee's operand is the first element of this argument.
            if (f.constant)
              f = Forwarded();
            ++f.level;
          }
        }
        if (f.param >= 0 && (f.level < 0 || f.level > 1))
          f = Forwarded();
        if (f.param >= 0 && f.level == 1)
          copy.elements.insert(key);
        if (f.param >= 0)
          copy.args[key] = static_cast<unsigned>(f.param);
        else if (f.receiver && std::string(key) == "stream")
          copy.receiver.insert(key);
        else if (f.constant && (std::string(key) == "kind" || std::string(key) == "stream"))
          copy.constants[key] = f.value;
        else if (f.constant && std::string(key) == "size" && f.value >= 0)
          copy.constants[key] = f.value;
        forwardsBuffer |= (std::string(key) == "dst" || std::string(key) == "src") && f.param >= 0;
        forwardsStream |= std::string(key) == "stream" && f.valid();
      }
      // A count matters only while an operand is still an array.
      if (copy.elements.empty()) {
        copy.args.erase("count");
        copy.constants.erase("count");
      }
      if (forwardsBuffer && forwardsStream)
        copies.push_back(copy);
    }
    result.copies = !copies.empty();
    bool agree = !copies.empty();
    auto withoutCount = [](cudalifetime::Model m) {
      m.args.erase("count");
      m.constants.erase("count");
      return m;
    };
    for (const auto &copy : copies) {
      auto a = withoutCount(copy), b = withoutCount(copies[0]);
      agree &= a.args == b.args && a.constants == b.constants && a.elements == b.elements && a.receiver == b.receiver;
    }
    if (agree && !result.synchronizes) {
      result.copy = copies[0];
      for (const auto &copy : copies)
        if (copy.args.count("count"))
          result.copy->args["count"] = copy.args.at("count");
    }
    memo[fd] = result;
    return memo[fd];
  }
  // A concrete byte count, and the bits it covers (0 when unknown or too large).
  static std::optional<uint64_t> byteCount(SVal size) {
    if (auto bytes = size.getAs<nonloc::ConcreteInt>())
      return bytes->getValue()->getLimitedValue();
    return std::nullopt;
  }
  static uint64_t bitsOf(std::optional<uint64_t> count) { return count && *count <= UINT64_MAX / 8 ? *count * 8 : 0; }
  // A concrete cudaMemcpyKind, or -1.
  static int kindOf(SVal kind) {
    if (auto v = kind.getAs<nonloc::ConcreteInt>())
      return v->getValue()->getSExtValue();
    return -1;
  }
  static SVal arg(ProgramStateRef s, const CallEvent &call, const cudalifetime::Model &m, const char *key) {
    auto &manager = s->getStateManager();
    if (auto constant = m.constants.find(key); constant != m.constants.end())
      return manager.getSValBuilder().makeIntVal(constant->second, manager.getContext().IntTy);
    if (m.receiver.count(key)) {
      // The operand is the receiver object, such as a stream wrapper's synchronize().
      const auto *instance = dyn_cast<CXXInstanceCall>(&call);
      const auto *method = instance ? dyn_cast_or_null<CXXMethodDecl>(instance->getDecl()) : nullptr;
      if (!method)
        return UnknownVal();
      SVal object = instance->getCXXThisVal();
      return handle(s, object, method->getThisType()->getPointeeType());
    }
    auto it = m.args.find(key);
    if (it == m.args.end() || it->second >= call.getNumArgs())
      return UnknownVal();
    SVal value = call.getArgSVal(it->second);
    const std::string k = key;
    if (m.elements.count(k)) {
      // An array of operands: the first entry. A count other than one leaves
      // the remaining entries unmodeled.
      auto count = m.args.count("count") && m.args.at("count") < call.getNumArgs()
                       ? call.getArgSVal(m.args.at("count")).getAs<nonloc::ConcreteInt>()
                       : std::nullopt;
      if (!count || count->getValue()->getZExtValue() != 1)
        cudalifetime::info().coverageUnknown.insert(
            "batched copy with a count other than one: entries after the first are not modeled");
      const auto *pointee = value.getAsRegion();
      const auto *type =
          call.getArgExpr(it->second) ? call.getArgExpr(it->second)->getType()->getAs<PointerType>() : nullptr;
      if (!pointee || !type)
        return UnknownVal();
      return s->getSVal(pointee, type->getPointeeType());
    }
    if ((k == "stream" || k == "event") && call.getArgExpr(it->second))
      value = handle(s, value, call.getArgExpr(it->second)->getType());
    return value;
  }
  // A stream or event passed as a wrapper object (rmm::cuda_stream_view,
  // cuda::stream_ref) is decoded to the handle it holds as its only data
  // member, so that it names the same stream as the raw handle.
  static SVal handle(ProgramStateRef s, SVal value, QualType type) {
    auto &manager = s->getStateManager();
    for (unsigned depth = 0; depth < 4; ++depth) {
      const auto *record = type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl();
      if (!record)
        return value;
      record = record->getDefinition();
      if (!record || record->getNumBases() || record->isPolymorphic())
        return UnknownVal();
      const FieldDecl *only = nullptr;
      unsigned fields = 0;
      for (const auto *field : record->fields()) {
        only = field;
        ++fields;
      }
      if (fields != 1)
        return UnknownVal();
      if (auto lazy = value.getAs<nonloc::LazyCompoundVal>()) {
        const auto *field = manager.getRegionManager().getFieldRegion(only, lazy->getRegion());
        value = manager.getStoreManager().getBinding(lazy->getStore(), loc::MemRegionVal(field), only->getType());
      } else if (const auto *object = dyn_cast_or_null<SubRegion>(value.getAsRegion())) {
        value = s->getSVal(manager.getRegionManager().getFieldRegion(only, object), only->getType());
      } else
        return UnknownVal();
      type = only->getType();
    }
    return UnknownVal();
  }
  static ProgramStateRef pending(ProgramStateRef s, const MemRegion *r, Stream stream, Access access,
                                 const Stmt *origin, CheckerContext &c, uint64_t bits = 0) {
    if (!r)
      return s;
    if (capturing(s, stream)) {
      noteCapture();
      return s;
    }
    return s->add<PendingOps>(
        {r, stream, sequence(s, stream), access, memory(s, r, bits), origin, c.getStackFrame(), bits});
  }

public:
  void checkLiveSymbols(ProgramStateRef s, SymbolReaper &reaper) const {
    for (auto entry : s->get<StreamValues>()) {
      if (auto symbol = entry.second.getAsSymbol())
        reaper.markLive(symbol);
      if (auto *region = entry.second.getAsRegion())
        reaper.markLive(region);
    }
    for (auto entry : s->get<EventValues>())
      if (auto symbol = entry.second.getAsSymbol())
        reaper.markLive(symbol);
    for (auto region : s->get<PreservedVariables>()) {
      reaper.markLive(region);
      if (auto symbol = s->getSVal(region).getAsSymbol())
        reaper.markLive(symbol);
    }
    for (auto entry : s->get<Memories>())
      reaper.markLive(entry.first);
    for (auto entry : s->get<ContainerOwners>()) {
      reaper.markLive(entry.first);
      reaper.markLive(entry.second);
    }
    for (auto entry : s->get<ViewResults>())
      reaper.markLive(entry.second);
    for (auto entry : s->get<Allocations>())
      reaper.markLive(entry.first);
    for (auto entry : s->get<PendingOps>())
      reaper.markLive(entry.region);
  }
  // CSA ends a path at a throw (it does not model exceptions), so a path that
  // can only end in a throw or a call that never returns changes nothing the
  // checker reports after it; end it where that becomes certain, before CSA
  // explores the message formatting and stack-trace collection of the error
  // branch (RAFT's exceptions walk the call stack in a loop).
  void checkBlockEntrance(const BlockEntrance &entrance, CheckerContext &c) const {
    const CFG *cfg = c.getStackFrame()->getCFG();
    const CFGBlock *block = entrance.getBlock();
    if (!cfg || !block)
      return;
    auto [it, inserted] = doomed.try_emplace(cfg);
    if (inserted)
      it->second = doomedBlocks(*cfg);
    if (block->getBlockID() < it->second.size() && it->second.test(block->getBlockID())) {
      ++cudalifetime::info().pathsEndedBeforeThrow;
      c.generateSink(c.getState(), c.getPredecessor());
    }
  }
  // Per top-level function: CSA may free the CFGs of one entry point before the next.
  mutable llvm::DenseMap<const CFG *, llvm::BitVector> doomed;
  mutable std::chrono::steady_clock::time_point started;
  void checkBeginFunction(CheckerContext &c) const {
    if (c.inTopFrame())
      started = std::chrono::steady_clock::now();
    // ExprEngine skips CUDAKernelCallExpr operand evaluation; keep referenced
    // locals alive so PostStmt can read the existing store without inventing a value.
    class References : public RecursiveASTVisitor<References> {
    public:
      std::vector<const VarDecl *> variables;
      bool VisitDeclRefExpr(DeclRefExpr *expr) {
        // Variables of a template pattern nested in the body (a generic
        // lambda's locals) have no storage in any frame, and reading a
        // constant one evaluates a value-dependent initializer.
        if (auto *decl = dyn_cast<VarDecl>(expr->getDecl()); decl && !decl->isTemplated())
          variables.push_back(decl);
        return true;
      }
    } references;
    const auto *decl = dyn_cast<FunctionDecl>(c.getStackFrame()->getDecl());
    if (!decl || !decl->hasBody())
      return;
    references.TraverseStmt(decl->getBody());
    auto s = c.getState();
    for (auto *variable : references.variables)
      s = s->add<PreservedVariables>(s->getLValue(variable, c.getStackFrame()).getAsRegion());
    c.addTransition(s);
  }
  void checkEndAnalysis(ExplodedGraph &graph, BugReporter &, ExprEngine &engine) const {
    auto &stats = cudalifetime::info();
    doomed.clear();
    ++stats.functionsAnalyzed;
    const Decl *decl = engine.getRootStackFrame()->getDecl();
    cudalifetime::EntryPoint entry;
    entry.function = functionName(decl);
    entry.nodes = graph.size();
    entry.milliseconds = static_cast<unsigned>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count());
    entry.exhausted = engine.hasWorkRemaining();
    if (entry.exhausted)
      stats.budgetExhausted.push_back(entry.function);
    if (stats.profileNodes)
      profile(graph, engine, entry);
    stats.entryPoints.push_back(std::move(entry));
  }
  static std::string functionName(const Decl *decl) {
    return isa<NamedDecl>(decl) ? cast<NamedDecl>(decl)->getQualifiedNameAsString() : "<anonymous>";
  }
  // Attribute the kept nodes to the function they are in, to the top-level
  // function's direct callee they descend from, and to their source line.
  static void profile(ExplodedGraph &graph, ExprEngine &engine, cudalifetime::EntryPoint &entry) {
    const StackFrame *root = engine.getRootStackFrame();
    const SourceManager &sources = engine.getContext().getSourceManager();
    llvm::DenseMap<const Decl *, unsigned> frames, callees;
    std::map<std::pair<FileID, unsigned>, unsigned> lines, sites;
    auto lineOf = [&](SourceLocation location) -> std::optional<std::pair<FileID, unsigned>> {
      auto [file, offset] = sources.getDecomposedExpansionLoc(location);
      if (!file.isValid())
        return std::nullopt;
      return std::make_pair(file, sources.getLineNumber(file, offset));
    };
    for (const ExplodedNode &node : graph.nodes()) {
      const StackFrame *frame = node.getStackFrame();
      if (!frame)
        continue;
      ++frames[frame->getDecl()];
      const StackFrame *callee = frame;
      while (callee->getParent() && callee->getParent() != root)
        callee = callee->getParent();
      ++callees[callee == root ? nullptr : callee->getDecl()];
      const Stmt *statement = node.getStmtForDiagnostics();
      if (statement)
        if (auto line = lineOf(statement->getBeginLoc()))
          ++lines[*line];
      // The project line the node is under: the node's own line in project
      // code, or the call site of the outermost library frame below it.
      const Stmt *site = statement;
      for (const StackFrame *f = frame; f && f->getParent(); f = f->getParent())
        if (sources.isInSystemHeader(sources.getExpansionLoc(f->getDecl()->getLocation())))
          site = f->getCallSite();
        else
          break;
      if (site)
        if (auto line = lineOf(site->getBeginLoc()))
          ++sites[*line];
    }
    auto top = [](auto &counts, auto label) {
      std::vector<std::pair<std::string, unsigned>> out;
      for (auto &[key, count] : counts)
        out.emplace_back(label(key), count);
      std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
      if (out.size() > 12)
        out.resize(12);
      return out;
    };
    entry.byFrame = top(frames, [](const Decl *d) { return functionName(d); });
    entry.byCallee = top(callees, [](const Decl *d) { return d ? functionName(d) : std::string("<the function itself>"); });
    auto label = [&](const std::pair<FileID, unsigned> &key) {
      auto entryFile = sources.getFileEntryRefForID(key.first);
      return (entryFile ? llvm::sys::path::filename(entryFile->getName()).str() : std::string("<unknown>")) + ":" +
             std::to_string(key.second);
    };
    entry.byLine = top(lines, label);
    entry.bySite = top(sites, label);
  }
  void checkPostCall(const CallEvent &call, CheckerContext &c) const {
    // Honor explicit no-return contracts inside inlined checked-return helpers.
    bool noReturn = false;
    if (const auto *decl = dyn_cast_or_null<FunctionDecl>(call.getDecl()))
      noReturn = decl->isNoReturn() || decl->hasAttr<AnalyzerNoReturnAttr>();
    if (const auto *expr = dyn_cast_or_null<CallExpr>(call.getOriginExpr()))
      if (!noReturn)
        noReturn = getFunctionExtInfo(expr->getCallee()->getType()).getNoReturn();
    if (noReturn) {
      c.generateSink(c.getState(), c.getPredecessor());
      return;
    }
    auto n = name(call);
    auto s = c.getState();
    // A call CSA evaluated conservatively (no body, or not inlined) may access
    // whatever reaches it; an inlined call's accesses were checked one by one.
    const bool conservative = model(n).effect.empty() && !c.wasInlined && !endsLifetime(call, n);
    if (conservative)
      s = opaqueCall(call, c, s);
    s = clearReach(s);
    if (n == "malloc" || n == "calloc" || n == "realloc")
      if (auto *region = base(call.getReturnValue()))
        s = s->set<Memories>(region, Pageable);
    s = trackContainers(call, s);
    // A body CSA declined to inline (a container method, or over an inlining
    // limit) still queues the copies its summary records.
    if (conservative)
      if (const auto *fd = dyn_cast_or_null<FunctionDecl>(call.getDecl()))
        if (fd->hasBody() && !cudalifetime::isCudaApi(fd))
          if (const auto &summary = summaryOf(fd); summary.copy || summary.sync) {
            const auto &applied = summary.copy ? *summary.copy : *summary.sync;
            cudalifetime::info().assumedModels.insert("generated summary: " + applied.function);
            applyCall(call, c, s, &applied);
            return;
          } else if (summary.through && summary.through->callee < call.getNumArgs()) {
            // The body polls the function its caller passes; the argument's
            // value names it (CSA replays a call without inlining when a loop
            // in it reaches the block visit limit, as RAFT's wait does).
            if (const FunctionDecl *target = call.getArgSVal(summary.through->callee).getAsFunctionDecl())
              if (auto resolved = resolveThrough(*summary.through, target)) {
                cudalifetime::info().assumedModels.insert("generated summary: " + fd->getQualifiedNameAsString() +
                                                          " waits through " + resolved->function);
                applyCall(call, c, s, &*resolved);
                return;
              }
          }
    if (s != c.getState())
      c.addTransition(s);
  }
  // Host containers own their buffer: a pointer data() returns is host memory
  // that the container's destructor releases. A view (span, string_view)
  // constructed from a container, or from another view, refers to the same
  // owner. Owners are std::vector, std::basic_string and thrust::host_vector,
  // and classes derived from them.
  static bool named(const CXXRecordDecl *record, const std::set<std::string> &names, unsigned depth = 0) {
    if (!record || depth > 8 || !(record = record->getDefinition()))
      return false;
    if (names.count(record->getQualifiedNameAsString()))
      return true;
    for (const auto &b : record->bases())
      if (named(b.getType()->getAsCXXRecordDecl(), names, depth + 1))
        return true;
    return false;
  }
  static bool hostContainer(QualType type) {
    static const std::set<std::string> names = {"std::vector", "std::basic_string", "std::__cxx11::basic_string",
                                                "thrust::host_vector"};
    return named(type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl(), names);
  }
  static bool hostView(QualType type) {
    static const std::set<std::string> names = {"std::span", "std::basic_string_view", "cudf::host_span"};
    return named(type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl(), names);
  }
  static const MemRegion *owner(ProgramStateRef s, SVal value, QualType type) {
    const MemRegion *object = value.getAsRegion();
    if (!object)
      if (auto lazy = value.getAs<nonloc::LazyCompoundVal>())
        object = lazy->getRegion();
    if (!object)
      return nullptr;
    object = object->StripCasts(); // a base-class subobject is the object itself
    if (hostContainer(type))
      return object;
    if (!hostView(type))
      return nullptr;
    if (const auto *known = s->get<ContainerOwners>(object))
      return *known;
    // A view copied without a constructor call (a trivial copy into a
    // parameter) is bound to the original's value.
    if (const auto *typed = dyn_cast<TypedValueRegion>(object))
      if (auto lazy = s->getSVal(typed).getAs<nonloc::LazyCompoundVal>())
        if (const auto *known = s->get<ContainerOwners>(lazy->getRegion()->StripCasts()))
          return *known;
    // A parameter initialized by such a call: look the call up by its argument,
    // through the casts and copies around it (host_span<T const>{c}).
    if (const auto *param = dyn_cast<ParamVarRegion>(object))
      if (const auto *origin = dyn_cast_or_null<CallExpr>(param->getOriginExpr());
          origin && !isa<CXXOperatorCallExpr>(origin) && param->getIndex() < origin->getNumArgs()) {
        const Expr *argument = origin->getArg(param->getIndex());
        for (unsigned depth = 0; argument && depth < 8; ++depth) {
          argument = argument->IgnoreImplicit()->IgnoreParens();
          if (const auto *known = s->get<ViewResults>(argument))
            return *known;
          if (const auto *cast = dyn_cast<ExplicitCastExpr>(argument))
            argument = cast->getSubExpr();
          else if (const auto *list = dyn_cast<InitListExpr>(argument); list && list->getNumInits() == 1)
            argument = list->getInit(0);
          else if (const auto *construct = dyn_cast<CXXConstructExpr>(argument);
                   construct && construct->getNumArgs() == 1)
            argument = construct->getArg(0);
          else
            break;
        }
      }
    return nullptr;
  }
  // Moving a container moves its buffer: the moved-to object owns it, so the
  // moved-from one releases nothing (a container returned by value, or
  // assigned from std::move).
  static ProgramStateRef transfer(ProgramStateRef s, SVal from, SVal to) {
    const auto *source = from.getAsRegion(), *target = to.getAsRegion();
    if (!source || !target)
      return s;
    source = source->StripCasts();
    target = target->StripCasts();
    for (auto entry : s->get<ContainerOwners>())
      if (entry.second == source)
        s = s->set<ContainerOwners>(entry.first, target);
    return s;
  }
  static ProgramStateRef trackContainers(const CallEvent &call, ProgramStateRef s) {
    if (const auto *ctor = dyn_cast<CXXConstructorCall>(&call);
        ctor && ctor->getDecl()->isMoveConstructor() && call.getNumArgs() == 1 &&
        hostContainer(ctor->getDecl()->getThisType()->getPointeeType()))
      return transfer(s, call.getArgSVal(0), ctor->getCXXThisVal());
    if (const auto *assign = dyn_cast<CXXMemberOperatorCall>(&call))
      if (const auto *method = dyn_cast_or_null<CXXMethodDecl>(assign->getDecl());
          method && method->isMoveAssignmentOperator() && call.getNumArgs() == 1 &&
          hostContainer(method->getThisType()->getPointeeType()))
        return transfer(s, call.getArgSVal(0), assign->getCXXThisVal());
    if (const auto *ctor = dyn_cast<CXXConstructorCall>(&call)) {
      if (hostView(ctor->getDecl()->getThisType()->getPointeeType()) && call.getNumArgs() >= 1 && call.getArgExpr(0))
        if (const auto *from = owner(s, call.getArgSVal(0), call.getArgExpr(0)->getType()))
          if (const auto *self = ctor->getCXXThisVal().getAsRegion())
            s = s->set<ContainerOwners>(self->StripCasts(), from);
      return s;
    }
    const auto *member = dyn_cast<CXXMemberCall>(&call);
    const auto *method = member ? dyn_cast_or_null<CXXMethodDecl>(member->getDecl()) : nullptr;
    if (!method)
      return s;
    const bool returnsView = hostView(call.getResultType()); // a conversion such as host_vector::operator host_span
    if (!returnsView &&
        (!method->getIdentifier() || method->getName() != "data" || !call.getResultType()->isPointerType()))
      return s;
    // data() is often inherited (span_base, vector_base): the receiver's own
    // type says whether it is a container or a view.
    QualType receiver = method->getThisType()->getPointeeType();
    if (const auto *expr = dyn_cast_or_null<CXXMemberCallExpr>(call.getOriginExpr()))
      if (const auto *object = expr->getImplicitObjectArgument()) {
        object = object->IgnoreParenBaseCasts();
        receiver = object->getType()->isPointerType() ? object->getType()->getPointeeType() : object->getType();
      }
    const auto *from = owner(s, member->getCXXThisVal(), receiver);
    if (returnsView) {
      const MemRegion *view = nullptr;
      if (auto constructed = call.getReturnValueUnderConstruction())
        view = constructed->getAsRegion();
      if (!view)
        if (auto lazy = call.getReturnValue().getAs<nonloc::LazyCompoundVal>())
          view = lazy->getRegion();
      if (!from)
        return s;
      if (view)
        return s->set<ContainerOwners>(view->StripCasts(), from);
      return call.getOriginExpr() ? s->set<ViewResults>(call.getOriginExpr(), from) : s;
    }
    const auto *buffer = base(call.getReturnValue());
    if (!from || !buffer)
      return s;
    s = s->set<ContainerOwners>(buffer, from);
    return memory(s, buffer) == Unknown ? s->set<Memories>(buffer, Pageable) : s;
  }
  static bool endsLifetime(const CallEvent &call, const std::string &n) {
    return isa<CXXDestructorCall>(call) || n == "free" || n == "operator delete" || n == "operator delete[]";
  }
  static bool pointsToConst(QualType type) {
    return (type->isPointerType() || type->isReferenceType()) && type->getPointeeType().isConstQualified();
  }
  ProgramStateRef opaqueCall(const CallEvent &call, CheckerContext &c, ProgramStateRef s) const;
  static ProgramStateRef reach(const CallEvent &call, ProgramStateRef s);
  static ProgramStateRef clearReach(ProgramStateRef s) {
    for (auto op : s->get<MutableReach>())
      s = s->remove<MutableReach>(op);
    for (auto op : s->get<ReadOnlyReach>())
      s = s->remove<ReadOnlyReach>(op);
    return s;
  }
  void hostAccess(CheckerContext &c, ProgramStateRef s, const MemRegion *r, uint64_t bits, bool isLoad,
                  const char *readMessage, const char *writeMessage) const {
    if (!r)
      return;
    for (auto op : s->get<PendingOps>()) {
      unsigned overlap = rangeOverlap(s, r, bits, op.region, op.bits);
      if (!overlap)
        continue;
      if ((isLoad && op.access == HostRead) || (!isLoad && (op.access == HostWrite || op.access == HostRead)))
        report(c, s, Rule::CL1, isLoad ? readMessage : writeMessage, op.memory, op.memory == Pinned ? "S1" : "S2", &op,
               overlap != 2);
    }
  }
  void checkPreCall(const CallEvent &call, CheckerContext &c) const;
  // Whether a call's status is consumed. A call in statement position (a block
  // item, the body of an if, loop, case or label, a for increment) discards it;
  // a condition, operand, argument, initializer or return value uses it.
  static bool resultIsUsed(const Expr *expr, ASTContext &context) {
    DynTypedNode current = DynTypedNode::create(*expr);
    const Stmt *child = expr;
    for (unsigned depth = 0; depth < 32; ++depth) {
      const auto &parents = context.getParents(current);
      if (parents.empty())
        return true;
      current = parents[0];
      if (current.get<CompoundStmt>() || current.get<SwitchCase>() || current.get<LabelStmt>() ||
          current.get<CXXForRangeStmt>())
        return false;
      if (const auto *statement = current.get<IfStmt>())
        return statement->getCond() == child;
      if (const auto *statement = current.get<WhileStmt>())
        return statement->getCond() == child;
      if (const auto *statement = current.get<DoStmt>())
        return statement->getCond() == child;
      if (const auto *statement = current.get<ForStmt>())
        return statement->getCond() == child;
      if (const auto *cast = current.get<CastExpr>()) {
        if (cast->getType()->isVoidType())
          return false;
        child = cast;
        continue;
      }
      if (current.get<ParenExpr>() || current.get<ExprWithCleanups>()) {
        child = current.get<Stmt>();
        continue;
      }
      return true;
    }
    return true;
  }
  bool evalCall(const CallEvent &call, CheckerContext &c) const;
  bool evalLibraryInternal(const CallEvent &call, CheckerContext &c) const;
  void applyCall(const CallEvent &call, CheckerContext &c, ProgramStateRef s,
                 const cudalifetime::Model *given = nullptr) const;
  // The call applyCall is handling, for the per-effect handlers below; each
  // returns the state to transition to.
  struct Site {
    const CallEvent &call;
    CheckerContext &c;
    const cudalifetime::Model &m;
    const std::string &name;
    Stream stream;
    const MemRegion *ptr;
  };
  void unmodeledCall(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applySyncCopy(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applyMemset(const Site &site, ProgramStateRef s, bool async) const;
  ProgramStateRef applyCaptureEnd(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applyAsyncCopy(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applyLaunch(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applyEventElapsed(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applyEventRecord(const Site &site, ProgramStateRef s) const;
  ProgramStateRef applyWait(const Site &site, ProgramStateRef s, bool hostSync) const;
  ProgramStateRef applyDeviceFree(const Site &site, ProgramStateRef s, bool async) const;
  ProgramStateRef applyHostRelease(const Site &site, ProgramStateRef s) const;
  void applyAllocation(const CallEvent &call, CheckerContext &c, ProgramStateRef s) const;
  void checkLocation(SVal location, bool isLoad, const Stmt *, CheckerContext &c) const;
  void checkBind(SVal location, SVal value, const Stmt *, bool, CheckerContext &c) const;
  void checkPostStmt(const CUDAKernelCallExpr *call, CheckerContext &c) const;
  void checkEndFunction(const ReturnStmt *, CheckerContext &c) const;
  ProgramStateRef checkPointerEscape(ProgramStateRef s, const InvalidatedSymbols &escaped, const CallEvent *call,
                                     PointerEscapeKind) const {
    // Escapes are counted for coverage only. They are not reported: on an
    // opaque call CSA also invalidates every global, so any buffer reachable
    // from a global would be reported at every such call. Reports use the
    // buffers reachable from the call's own arguments (reach, opaqueCall).
    ++cudalifetime::info().escapes;
    // CSA also emits escape notifications while processing summarized runtime
    // calls. Their pointer effects are handled above, not opaque ownership.
    if (call && !model(name(*call)).effect.empty())
      return s;
    bool pendingEscape = false;
    for (auto op : s->get<PendingOps>()) {
      if (const auto *symbolic = op.region->getSymbolicBase())
        pendingEscape |= escaped.count(symbolic->getSymbol()) != 0;
      if (call) {
        for (unsigned i = 0; i < call->getNumArgs(); ++i)
          pendingEscape |= overlaps(s, base(call->getArgSVal(i)), op.region);
        if (const auto *method = dyn_cast<CXXInstanceCall>(call))
          pendingEscape |= overlaps(s, base(method->getCXXThisVal()), op.region);
      }
    }
    if (pendingEscape) {
      ++cudalifetime::info().pendingPointerEscapes;
      cudalifetime::info().coverageUnknown.insert(
          "pending region escapes: subsequent accesses through opaque ownership are not modeled");
    }
    return s; // Escape cannot establish completion or retire the operation.
  }
};

void AsyncLifetimeChecker::checkPreCall(const CallEvent &call, CheckerContext &c) const {
  if (model(name(call)).effect.empty())
    applyCall(call, c, c.getState());
}

namespace {
// Symbols and regions reachable from a value through memory, as of PreCall.
struct Reachable : SymbolVisitor {
  llvm::SmallPtrSet<SymbolRef, 16> symbols;
  llvm::SmallPtrSet<const MemRegion *, 16> regions;
  bool VisitSymbol(SymbolRef symbol) override {
    symbols.insert(symbol);
    return true;
  }
  bool VisitMemRegion(const MemRegion *region) override {
    regions.insert(region);
    return true;
  }
  bool contains(const MemRegion *r) const {
    if (const auto *symbolic = r->getSymbolicBase(); symbolic && symbols.count(symbolic->getSymbol()))
      return true;
    for (const auto *region : regions)
      if (region == r || r->isSubRegionOf(region) || region->isSubRegionOf(r))
        return true;
    return false;
  }
};
} // namespace

// Which pending buffers the call's arguments reach, and whether the callee may
// write them: directly through a parameter (writable unless it points to const),
// through the receiver (unless the method is const), or through pointers stored
// in memory an argument reaches (always writable).
ProgramStateRef AsyncLifetimeChecker::reach(const CallEvent &call, ProgramStateRef s) {
  s = clearReach(s);
  if (s->get<PendingOps>().isEmpty())
    return s;
  const auto parameters = call.parameters();
  std::vector<std::pair<const MemRegion *, bool>> direct; // region, writable
  Reachable indirect;
  for (unsigned i = 0; i < call.getNumArgs(); ++i) {
    const SVal value = call.getArgSVal(i);
    // Variadic and non-pointer parameters give the callee unknown access.
    if (auto *r = base(value))
      direct.push_back({r, !(i < parameters.size() && pointsToConst(parameters[i]->getType()))});
    s->scanReachableSymbols(value, indirect);
  }
  if (const auto *method = dyn_cast<CXXInstanceCall>(&call)) {
    const SVal value = method->getCXXThisVal();
    const auto *decl = dyn_cast_or_null<CXXMethodDecl>(call.getDecl());
    if (auto *r = base(value))
      direct.push_back({r, !(decl && decl->isConst())});
    s->scanReachableSymbols(value, indirect);
  }
  for (auto op : s->get<PendingOps>()) {
    bool reaches = false, writable = false, viaArgument = false;
    for (const auto &[r, w] : direct)
      if (overlaps(s, r, op.region)) {
        reaches = viaArgument = true;
        writable |= w;
      }
    if (!viaArgument && indirect.contains(op.region))
      reaches = writable = true;
    if (reaches)
      s = writable ? s->add<MutableReach>(op) : s->add<ReadOnlyReach>(op);
  }
  return s;
}

ProgramStateRef AsyncLifetimeChecker::opaqueCall(const CallEvent &call, CheckerContext &c, ProgramStateRef s) const {
  const auto writable = s->get<MutableReach>();
  const auto readOnly = s->get<ReadOnlyReach>();
  s = clearReach(s);
  bool argumentUse = !writable.isEmpty() || !readOnly.isEmpty();
  cudalifetime::info().pendingArgumentUses += argumentUse;
  if (const auto *method = dyn_cast<CXXInstanceCall>(&call))
    for (auto op : s->get<PendingOps>())
      if (overlaps(s, base(method->getCXXThisVal()), op.region)) {
        ++cudalifetime::info().pendingMemberCallUses;
        break;
      }
  auto check = [&](const Pending &op, bool mayWrite) {
    if (op.access == DeviceUse) {
      cudalifetime::info().coverageUnknown.insert(
          "pending device region reaches an opaque call: device access effects are unknown");
      return;
    }
    // Through pointers to const the callee can only read, which conflicts
    // only with a pending D2H into the buffer.
    if (mayWrite)
      report(c, s, Rule::CL1,
             "Host buffer still used by asynchronous work is passed to an opaque call that may access it", op.memory,
             "S2", &op, true);
    else if (op.access == HostRead)
      report(c, s, Rule::CL1, "Opaque call may read a host buffer before asynchronous D2H completion", op.memory, "S2",
             &op, true);
  };
  for (auto op : writable)
    check(op, true);
  for (auto op : readOnly)
    if (!writable.contains(op))
      check(op, false);
  return s;
}

void AsyncLifetimeChecker::applyCall(const CallEvent &call, CheckerContext &c, ProgramStateRef s,
                                     const cudalifetime::Model *given) const {
  const std::string n = name(call);
  const cudalifetime::Model m = given ? *given : model(n);
  const Effect e = effectOf(m.effect);
  const Stream stream = streamIdentity(s, arg(s, call, m, "stream"));
  if (m.args.count("stream") && stream == 0)
    ++cudalifetime::info().unmodeled[n + ":unknown-stream"];
  const Site site{call, c, m, n, stream, base(arg(s, call, m, "ptr"))};
  if (e == Effect::None) {
    unmodeledCall(site, s);
    return;
  }
  ++cudalifetime::info().modeledCalls;
  const bool queues =
      e == Effect::EnqueueCopy || e == Effect::Launch || e == Effect::MemsetAsync || e == Effect::FreeAsync;
  if (queues && capturing(s, stream)) {
    noteCapture(); // recorded into the graph; nothing runs until the graph is launched
    c.addTransition(s);
    return;
  }
  switch (e) {
  case Effect::Copy:
    s = applySyncCopy(site, s);
    break;
  case Effect::Memset:
    s = applyMemset(site, s, false);
    break;
  case Effect::MemsetAsync:
    s = applyMemset(site, s, true);
    break;
  case Effect::CaptureBegin:
    s = s->set<CaptureOrigins>(stream, stream);
    noteCapture();
    break;
  case Effect::CaptureEnd:
    s = applyCaptureEnd(site, s);
    break;
  case Effect::GraphLaunch:
    s = enqueue(s, stream, arg(s, call, m, "stream"));
    noteCapture();
    break;
  case Effect::EnqueueCopy:
    s = applyAsyncCopy(site, s);
    break;
  case Effect::Launch:
    s = applyLaunch(site, s);
    break;
  case Effect::SyncStream:
    s = syncStream(s, stream);
    break;
  case Effect::SyncDevice:
    for (auto seq : s->get<Sequences>())
      s = retire(s, seq.first, seq.second);
    break;
  case Effect::EventElapsed:
    s = applyEventElapsed(site, s);
    break;
  case Effect::EventRecord:
    s = applyEventRecord(site, s);
    break;
  case Effect::StreamWaitEvent:
    s = applyWait(site, s, false);
    break;
  case Effect::SyncEvent:
    s = applyWait(site, s, true);
    break;
  case Effect::FreeAsync:
    s = applyDeviceFree(site, s, true);
    break;
  case Effect::FreeDevice:
    s = applyDeviceFree(site, s, false);
    break;
  case Effect::FreeHost:
  case Effect::HostUnregister:
    s = applyHostRelease(site, s);
    break;
  default:
    break; // no stream or buffer effect (destroy, or an effect handled elsewhere)
  }
  c.addTransition(s);
}

// A call with no model. A lifetime ends here whether or not the callee is
// inlined; other calls are checked at PostCall, once it is known whether CSA
// inlined them.
void AsyncLifetimeChecker::unmodeledCall(const Site &site, ProgramStateRef s) const {
  const CallEvent &call = site.call;
  CheckerContext &c = site.c;
  if (cudalifetime::isCudaApi(dyn_cast_or_null<FunctionDecl>(call.getDecl())))
    ++cudalifetime::info().unmodeled[site.name];
  if (!endsLifetime(call, site.name)) {
    auto next = reach(call, s);
    if (next != s)
      c.addTransition(next);
    return;
  }
  const bool destructor = isa<CXXDestructorCall>(call);
  const MemRegion *self = destructor ? cast<CXXDestructorCall>(call).getCXXThisVal().getAsRegion() : nullptr;
  if (self)
    self = self->StripCasts();
  for (auto op : s->get<PendingOps>()) {
    bool match = false;
    for (unsigned i = 0; i < call.getNumArgs(); ++i)
      match |= overlaps(s, base(call.getArgSVal(i)), op.region);
    if (const auto *method = dyn_cast<CXXInstanceCall>(&call))
      match |= overlaps(s, base(method->getCXXThisVal()), op.region);
    const MemRegion *const *owned = self ? s->get<ContainerOwners>(op.region->getBaseRegion()) : nullptr;
    if (owned && *owned == self && op.access != DeviceUse) {
      report(c, s, Rule::CL2,
             "Host container is destroyed, releasing its buffer, while asynchronous work may still use it", op.memory,
             "S4", &op, true);
      continue;
    }
    if (match && op.access != DeviceUse)
      report(c, s, Rule::CL2,
             destructor ? "Object lifetime ends with pending asynchronous work"
                        : "Host buffer is released with pending asynchronous work",
             op.memory, "S4", &op, true);
  }
}

// Synchronous cudaMemcpy runs on the default stream. With host memory on
// either side it returns only after the copy and all earlier work on the
// default stream (and on the streams it waits for) is complete, or, for a
// pageable H2D source, once the source is staged; device-to-device copies
// do not synchronize. CUDA 13.2.2 archive: api-sync-behavior#memcpy.
ProgramStateRef AsyncLifetimeChecker::applySyncCopy(const Site &site, ProgramStateRef s) const {
  const CallEvent &call = site.call;
  CheckerContext &c = site.c;
  const Stream def = defaultStream();
  auto *dst = base(arg(s, call, site.m, "dst")), *src = base(arg(s, call, site.m, "src"));
  const uint64_t bits = bitsOf(byteCount(arg(s, call, site.m, "size")));
  const int copyKind = kindOf(arg(s, call, site.m, "kind"));
  s = enqueue(s, def, c.getSValBuilder().makeZeroVal(c.getASTContext().VoidPtrTy));
  deviceUse(c, s, src, def, "cudaMemcpy");
  deviceUse(c, s, dst, def, "cudaMemcpy");
  auto host = [&](const MemRegion *r) {
    const Memory k = memory(s, r);
    return k == Pinned || k == Pageable;
  };
  // With cudaMemcpyDefault and no operand known to be host memory, the copy
  // is treated as device to device; say so when an operand's kind is unknown.
  if (copyKind == DefaultCopy && !host(dst) && !host(src) && (memory(s, dst) == Unknown || memory(s, src) == Unknown))
    unknownDirection();
  const bool writesHost = copyKind == HostToHost || copyKind == DeviceToHost || (copyKind == DefaultCopy && host(dst));
  const bool readsHost = copyKind == HostToHost || copyKind == HostToDevice || (copyKind == DefaultCopy && host(src));
  if (writesHost || readsHost) {
    s = syncStream(s, def);
    // What remains pending is on streams the default stream does not wait for.
    if (writesHost)
      hostAccess(c, s, dst, bits, false, "",
                 "cudaMemcpy writes a host buffer that asynchronous work on another stream still uses");
    if (readsHost)
      hostAccess(c, s, src, bits, true,
                 "cudaMemcpy reads a host buffer before an asynchronous D2H on another stream completes", "");
  } else if (copyKind == DeviceToDevice || copyKind == DefaultCopy) {
    s = pending(s, dst, def, DeviceUse, call.getOriginExpr(), c);
    s = pending(s, src, def, DeviceUse, call.getOriginExpr(), c);
  } else
    ++cudalifetime::info().unmodeled[site.name + ":unknown-copy-kind"];
  return s;
}

// cudaMemset is asynchronous with respect to the host unless the target is
// pinned host memory. CUDA 13.2.2 archive: api-sync-behavior#memset.
ProgramStateRef AsyncLifetimeChecker::applyMemset(const Site &site, ProgramStateRef s, bool async) const {
  const CallEvent &call = site.call;
  CheckerContext &c = site.c;
  const Stream target = async ? site.stream : defaultStream();
  const uint64_t bits = bitsOf(byteCount(arg(s, call, site.m, "size")));
  s = enqueue(s, target,
              async ? arg(s, call, site.m, "stream") : c.getSValBuilder().makeZeroVal(c.getASTContext().VoidPtrTy));
  deviceUse(c, s, site.ptr, target, async ? "cudaMemsetAsync" : "cudaMemset");
  const bool pinned = memory(s, site.ptr, bits) == Pinned;
  if (!async && pinned) {
    s = syncStream(s, target);
    hostAccess(c, s, site.ptr, bits, false, "",
               "cudaMemset writes a host buffer that asynchronous work on another stream still uses");
  } else
    s = pending(s, site.ptr, target, pinned ? HostRead : DeviceUse, call.getOriginExpr(), c, bits);
  return s;
}

ProgramStateRef AsyncLifetimeChecker::applyCaptureEnd(const Site &site, ProgramStateRef s) const {
  for (auto entry : s->get<CaptureOrigins>())
    if (entry.second == site.stream)
      s = s->remove<CaptureOrigins>(entry.first);
  for (auto entry : s->get<CapturedEvents>())
    if (entry.second == site.stream)
      s = s->remove<CapturedEvents>(entry.first);
  if (auto out = arg(s, site.call, site.m, "ptr").getAs<Loc>())
    s = s->bindLoc(*out, UnknownVal(), site.c.getStackFrame());
  return s;
}

ProgramStateRef AsyncLifetimeChecker::applyAsyncCopy(const Site &site, ProgramStateRef s) const {
  const CallEvent &call = site.call;
  CheckerContext &c = site.c;
  const Stream stream = site.stream;
  s = enqueue(s, stream, arg(s, call, site.m, "stream"));
  auto *dst = base(arg(s, call, site.m, "dst")), *src = base(arg(s, call, site.m, "src"));
  const std::optional<uint64_t> count = byteCount(arg(s, call, site.m, "size"));
  if (count && *count == 0)
    return s; // an empty copy touches nothing
  const uint64_t bits = bitsOf(count);
  const int copyKind = kindOf(arg(s, call, site.m, "kind"));
  const Stmt *origin = call.getOriginExpr();
  deviceUse(c, s, src, stream, "cudaMemcpyAsync");
  deviceUse(c, s, dst, stream, "cudaMemcpyAsync");
  switch (copyKind) {
  case HostToDevice:
    s = pending(s, src, stream, HostWrite, origin, c, bits);
    s = pending(s, dst, stream, DeviceUse, origin, c);
    break;
  case DeviceToHost:
    s = pending(s, dst, stream, HostRead, origin, c, bits);
    s = pending(s, src, stream, DeviceUse, origin, c);
    break;
  case DeviceToDevice:
    s = pending(s, dst, stream, DeviceUse, origin, c);
    s = pending(s, src, stream, DeviceUse, origin, c);
    break;
  case DefaultCopy: {
    // cudaMemcpyDefault: unified addressing picks the direction from the
    // pointers. A host destination is written (D2H or H2H), so the host must
    // not touch it; a host source is read (H2D or H2H), so the host must not
    // write it. An operand of unknown memory kind is left out and reported.
    const Memory dstKind = memory(s, dst, bits), srcKind = memory(s, src, bits);
    if (dstKind == Pinned || dstKind == Pageable)
      s = pending(s, dst, stream, HostRead, origin, c, bits);
    else if (dstKind == Device || dstKind == Ordered)
      s = pending(s, dst, stream, DeviceUse, origin, c);
    if (srcKind == Pinned || srcKind == Pageable)
      s = pending(s, src, stream, HostWrite, origin, c, bits);
    else if (srcKind == Device || srcKind == Ordered)
      s = pending(s, src, stream, DeviceUse, origin, c);
    if (dstKind == Unknown || srcKind == Unknown)
      unknownDirection();
    break;
  }
  case HostToHost:
    break;
  default:
    ++cudalifetime::info().unmodeled[site.name + ":unknown-copy-kind"];
    break;
  }
  return s;
}

ProgramStateRef AsyncLifetimeChecker::applyLaunch(const Site &site, ProgramStateRef s) const {
  const CallEvent &call = site.call;
  s = enqueue(s, site.stream, arg(s, call, site.m, "stream"));
  for (unsigned i = 0; i < call.getNumArgs(); ++i) {
    if (site.m.args.count("stream") && site.m.args.at("stream") == i)
      continue;
    auto *buffer = base(call.getArgSVal(i));
    deviceUse(site.c, s, buffer, site.stream, "Modeled launch");
    s = pending(s, buffer, site.stream, DeviceUse, call.getOriginExpr(), site.c);
  }
  return s;
}

// CUDA 13.2.2 event API: an incomplete start or end returns NotReady. A
// successful timing query therefore proves only their recorded captures.
ProgramStateRef AsyncLifetimeChecker::applyEventElapsed(const Site &site, ProgramStateRef s) const {
  for (const char *key : {"start", "end"}) {
    const uint64_t event = eventIdentity(s, arg(s, site.call, site.m, key));
    for (auto b : s->get<EventBounds>())
      if (b.first.first == event && b.first.second)
        s = retire(s, b.first.second, b.second);
  }
  if (auto out = arg(s, site.call, site.m, "ms").getAs<Loc>())
    s = s->bindLoc(*out, UnknownVal(), site.c.getStackFrame());
  return s;
}

ProgramStateRef AsyncLifetimeChecker::applyEventRecord(const Site &site, ProgramStateRef s) const {
  const Stream stream = site.stream;
  const SVal value = arg(s, site.call, site.m, "event");
  const uint64_t event = eventIdentity(s, value);
  s = s->set<EventValues>(event, value);
  if (capturing(s, stream)) {
    // A record inside a capture is a graph node; a stream that waits on it
    // joins the capture.
    for (auto b : s->get<EventBounds>())
      if (b.first.first == event)
        s = s->remove<EventBounds>(b.first);
    return s->set<RecordedEvents>(event, true)->set<CapturedEvents>(event, *s->get<CaptureOrigins>(stream));
  }
  s = enqueue(s, stream, arg(s, site.call, site.m, "stream"));
  for (auto b : s->get<EventBounds>())
    if (b.first.first == event)
      s = s->remove<EventBounds>(b.first);
  s = s->set<RecordedEvents>(event, true)->set<EventBounds>({event, stream}, sequence(s, stream));
  for (auto wait : s->get<WaitBounds>())
    if (wait.first.first == stream)
      s = s->set<EventBounds>({event, wait.first.second}, wait.second);
  return s;
}

// cudaStreamWaitEvent orders the stream after the event's capture;
// cudaEventSynchronize (hostSync) completes that capture for the host.
ProgramStateRef AsyncLifetimeChecker::applyWait(const Site &site, ProgramStateRef s, bool hostSync) const {
  const CallEvent &call = site.call;
  CheckerContext &c = site.c;
  const Stream stream = site.stream;
  const uint64_t event = eventIdentity(s, arg(s, call, site.m, "event"));
  if (!hostSync && (capturing(s, stream) || s->get<CapturedEvents>(event))) {
    if (const auto *origin = s->get<CapturedEvents>(event))
      s = s->set<CaptureOrigins>(stream, *origin);
    noteCapture();
    return s;
  }
  s = s->set<EventValues>(event, arg(s, call, site.m, "event"));
  const auto *recorded = s->get<RecordedEvents>(event);
  // Only an event created on this path is known to be unrecorded; one read
  // from an object, a container or another process has unknown history.
  if (recorded && !*recorded)
    report(c, s, Rule::CL4, "Event has no recorded capture on this path; waiting establishes no dependency", Unknown,
           "S8", nullptr, true);
  else if (!recorded)
    cudalifetime::info().coverageUnknown.insert(
        "event with no creation or record on this path: the dependency it carries is unknown");
  if (!hostSync)
    s = enqueue(s, stream, arg(s, call, site.m, "stream"));
  for (auto b : s->get<EventBounds>())
    if (b.first.first == event) {
      if (hostSync && b.first.second)
        s = retire(s, b.first.second, b.second);
      else
        s = bound(s, stream, b.first.second, b.second);
      // Work after the capture matters only if the waiter is not already ordered after it.
      for (auto alloc : s->get<Allocations>())
        if (!alloc.second.freed && alloc.second.stream == b.first.second && alloc.second.seq > b.second &&
            !ordered(s, stream, alloc.second.stream, alloc.second.seq))
          report(c, s, Rule::CL4, "Event capture predates asynchronous allocation and does not cover it", Ordered, "S8",
                 nullptr, true);
      for (auto op : s->get<PendingOps>())
        if (op.stream == b.first.second && op.seq > b.second && !ordered(s, stream, op.stream, op.seq))
          report(c, s, Rule::CL4, "Event capture predates pending producer work and does not cover it", op.memory, "S8",
                 &op, true);
    }
  return s;
}

// cudaFreeAsync (async) and cudaFree of device memory. Ordinary cudaFree
// does not prove unrelated host completion.
ProgramStateRef AsyncLifetimeChecker::applyDeviceFree(const Site &site, ProgramStateRef s, bool async) const {
  CheckerContext &c = site.c;
  const Stream stream = site.stream;
  const MemRegion *r = site.ptr;
  if (async)
    s = enqueue(s, stream, arg(s, site.call, site.m, "stream"));
  if (const auto *key = allocationRegion(s, r)) {
    const auto *alloc = s->get<Allocations>(key);
    bool reported = false;
    for (auto op : s->get<PendingOps>())
      if (overlaps(s, op.region, r)) {
        bool safe = async ? ordered(s, stream, op.stream, op.seq) : completed(s, op.stream) >= op.seq;
        if (!safe) {
          report(c, s, Rule::CL3, "Stream-ordered allocation is freed while another operation may still use it",
                 Ordered, async ? "S6" : "S7", &op, async && !distinct(s, stream, op.stream));
          reported = true;
          break;
        }
      }
    if (!reported && async && !ordered(s, stream, alloc->stream, alloc->seq))
      report(c, s, Rule::CL3, "cudaFreeAsync has no dependency on the asynchronous allocation", Ordered, "S6", nullptr,
             !distinct(s, stream, alloc->stream), false, alloc, key);
    if (!reported && !async && completed(s, alloc->stream) < alloc->seq)
      report(c, s, Rule::CL3, "Asynchronous allocation is not known complete before cudaFree", Ordered, "S7", nullptr,
             false, false, alloc, key);
    auto allocation = *alloc;
    allocation.freed = true;
    allocation.freeStream = stream;
    allocation.freeSeq = sequence(s, stream);
    s = s->set<Allocations>(key, allocation);
  } else if (!async) {
    for (auto op : s->get<PendingOps>())
      if (op.access == DeviceUse && overlaps(s, op.region, r)) {
        report(c, s, Rule::CL3, "Ordinary cudaFree may implicitly synchronize; this path relies on that behavior",
               Device, "S10", &op, false, true);
        break;
      }
  }
  return s;
}

// cudaFreeHost and cudaHostUnregister.
ProgramStateRef AsyncLifetimeChecker::applyHostRelease(const Site &site, ProgramStateRef s) const {
  const MemRegion *r = site.ptr;
  for (auto op : s->get<PendingOps>())
    if (overlaps(s, op.region, r) && op.access != DeviceUse)
      report(site.c, s, Rule::CL2, "Host buffer is released or unregistered before asynchronous work is known complete",
             op.memory, "S4", &op, true);
  if (r)
    s = s->remove<Memories>(r)->remove<RegisteredExtents>(r);
  return s;
}

void AsyncLifetimeChecker::applyAllocation(const CallEvent &call, CheckerContext &c, ProgramStateRef s) const {
  auto m = model(name(call));
  const Effect e = effectOf(m.effect);
  if (e == Effect::HostRegister) {
    if (auto *r = base(arg(s, call, m, "ptr"))) {
      const uint64_t bits = bitsOf(byteCount(arg(s, call, m, "size")));
      c.addTransition(s->set<Memories>(r, Pinned)->set<RegisteredExtents>(r, bits));
    }
    return;
  }
  if (e != Effect::MallocHost && e != Effect::MallocDevice && e != Effect::MallocAsync && e != Effect::StreamCreate &&
      e != Effect::EventCreate)
    return;
  auto out = arg(s, call, m, "ptr").getAs<Loc>();
  if (!out)
    return;
  SVal value = UnknownVal();
  {
    auto *origin = call.getOriginExpr();
    if (!origin)
      return;
    QualType type = c.getASTContext().VoidPtrTy;
    value = c.getSValBuilder().conjureSymbolVal(call, type, c.blockCount());
    s = s->bindLoc(*out, value, c.getStackFrame());
  }
  if (e == Effect::StreamCreate) {
    StreamKind streamKind = name(call) == "cudaStreamCreate" ? StreamKind::Blocking : StreamKind::Unknown;
    if (auto v = arg(s, call, m, "flags").getAs<nonloc::ConcreteInt>())
      streamKind =
          (v->getValue()->getZExtValue() & StreamNonBlockingFlag) ? StreamKind::NonBlocking : StreamKind::Blocking;
    s = s->set<StreamKinds>(identity(value), static_cast<unsigned>(streamKind))
            ->set<StreamValues>(identity(value), value);
  } else if (e == Effect::EventCreate) {
    s = s->set<RecordedEvents>(identity(value), false)->set<EventValues>(identity(value), value);
    if (auto flags = arg(s, call, m, "flags").getAs<nonloc::ConcreteInt>())
      s = s->set<EventTimingDisabled>(identity(value),
                                      (flags->getValue()->getZExtValue() & EventDisableTimingFlag) != 0);
  } else if (auto *r = base(value)) {
    Memory mem = e == Effect::MallocHost ? Pinned : e == Effect::MallocAsync ? Ordered : Device;
    s = s->set<Memories>(r, mem);
    if (mem == Ordered) {
      Stream stream = streamIdentity(s, arg(s, call, m, "stream"));
      s = enqueue(s, stream, arg(s, call, m, "stream"));
      s = s->set<Allocations>(r, {stream, sequence(s, stream), false, 0, 0, call.getOriginExpr(), c.getStackFrame()});
    }
  }
  c.addTransition(s);
}

// Standard-library internals that cannot reach CUDA and whose branches turn
// on values no path determines: shared_ptr's reference counts (each copy and
// destruction splits on the count and on whether the program is threaded),
// iostream state, and locking. CSA inlines them by default; they are
// evaluated as CSA evaluates a call it does not inline (arguments and globals
// invalidated, a fresh result), unless a pending buffer reaches the call.
// The deleter a last reference runs is not followed.
static bool libraryInternal(const FunctionDecl *fd, const SourceManager &sources) {
  if (!fd || isa<CXXConstructorDecl>(fd) || isa<CXXDestructorDecl>(fd) ||
      !sources.isInSystemHeader(sources.getExpansionLoc(fd->getLocation())))
    return false;
  const QualType result = fd->getReturnType();
  if (!result->isVoidType() && !result->isScalarType() && !result->isReferenceType())
    return false;
  const DeclContext *context = fd->getDeclContext();
  std::string owner = isa<CXXRecordDecl>(context) ? cast<CXXRecordDecl>(context)->getQualifiedNameAsString() : "";
  // A free stream operator (operator<<(ostream &, ...)) belongs to its stream.
  if (owner.empty() && fd->getNumParams() && fd->getParamDecl(0)->getType()->isReferenceType())
    if (const auto *record = fd->getParamDecl(0)->getType().getNonReferenceType()->getAsCXXRecordDecl())
      owner = record->getQualifiedNameAsString();
  const std::string function = fd->getQualifiedNameAsString();
  static const char *const owners[] = {"std::_Sp_counted_base", "std::__shared_weak_count", "std::__shared_count",
                                       "std::basic_ostream", "std::basic_istream", "std::basic_iostream",
                                       "std::basic_ios", "std::ios_base", "std::basic_streambuf",
                                       "std::basic_stringbuf", "std::mutex", "std::recursive_mutex",
                                       "std::__mutex_base", "std::unique_lock"};
  static const char *const functions[] = {"__gnu_cxx::__exchange_and_add_dispatch", "__gnu_cxx::__atomic_add_dispatch",
                                          "__gnu_cxx::__is_single_threaded", "std::endl", "std::flush"};
  auto prefixed = [](const std::string &name, const char *prefix) {
    const size_t n = std::char_traits<char>::length(prefix);
    return name.compare(0, n, prefix) == 0 && (name.size() == n || name[n] == ':' || name[n] == '<');
  };
  for (const char *prefix : owners)
    if (prefixed(owner, prefix))
      return true;
  for (const char *prefix : functions)
    if (prefixed(function, prefix))
      return true;
  return false;
}

bool AsyncLifetimeChecker::evalLibraryInternal(const CallEvent &call, CheckerContext &c) const {
  const auto *fd = dyn_cast_or_null<FunctionDecl>(call.getDecl());
  if (!libraryInternal(fd, c.getSourceManager()))
    return false;
  ProgramStateRef s = reach(call, c.getState());
  const bool reached = !s->get<MutableReach>().isEmpty() || !s->get<ReadOnlyReach>().isEmpty();
  if (reached)
    return false;
  s = call.invalidateRegions(c.blockCount(), clearReach(s));
  const QualType result = call.getResultType();
  if (!result->isVoidType())
    s = s->BindExpr(call.getOriginExpr(), c.getStackFrame(),
                    c.getSValBuilder().conjureSymbolVal(call, result, c.blockCount(), this));
  ++cudalifetime::info().libraryCallsNotInlined;
  c.addTransition(s);
  return true;
}

bool AsyncLifetimeChecker::evalCall(const CallEvent &call, CheckerContext &c) const {
  cudalifetime::info().perThread = c.getPreprocessor().isMacroDefined("CUDA_API_PER_THREAD_DEFAULT_STREAM");
  const auto m = model(name(call));
  const Effect effect = effectOf(m.effect);
  if (effect == Effect::None && call.getOriginExpr() && evalLibraryInternal(call, c))
    return true;
  if (effect == Effect::None || effect == Effect::Observe || !call.getOriginExpr())
    return false;
  if (effect == Effect::AddressOnly) { // inspects the pointer value, not the buffer: CSA evaluates it
    if (m.assumption)
      cudalifetime::info().assumedModels.insert(m.function);
    return false;
  }
  for (const auto &[key, index] : m.args)
    if (index >= call.getNumArgs()) {
      if (key == "flags" && name(call) == "cudaStreamCreate")
        continue;
      ++cudalifetime::info().unmodeled[m.function + ":model-argument-out-of-range"];
      return false;
    }
  if (m.assumption)
    if (const auto *fd = dyn_cast_or_null<FunctionDecl>(call.getDecl()))
      if (fd->hasBody()) {
        ++cudalifetime::info().unmodeled[m.function + ":model-body-deferred-to-csa"];
        return false;
      }
  if (m.assumption)
    cudalifetime::info().assumedModels.insert(m.function);
  auto s = c.getState();
  bool impossibleSuccess = false;
  if (effect == Effect::EventElapsed)
    for (const char *key : {"start", "end"}) {
      const uint64_t event = eventIdentity(s, arg(s, call, m, key));
      const auto *recorded = s->get<RecordedEvents>(event);
      const auto *disabled = s->get<EventTimingDisabled>(event);
      impossibleSuccess |= (recorded && !*recorded) || (disabled && *disabled);
    }
  auto &builder = c.getSValBuilder();
  QualType resultType = call.getResultType();
  if (resultType->isIntegralOrEnumerationType()) {
    auto result = builder.conjureSymbolVal(call, resultType, c.blockCount(), this);
    auto failed = s->assume(result, true);
    if (failed) {
      failed = failed->BindExpr(call.getOriginExpr(), c.getStackFrame(), result);
      if (resultIsUsed(call.getOriginExpr(), c.getASTContext()) || name(call) == "cudaEventQuery" ||
          name(call) == "cudaStreamQuery" || effect == Effect::EventElapsed)
        c.addTransition(failed);
      else {
        ++cudalifetime::info().cudaFailurePathsSkipped;
        c.generateSink(failed, c.getPredecessor());
      }
    }
    if (impossibleSuccess)
      return true;
    s = s->BindExpr(call.getOriginExpr(), c.getStackFrame(), builder.makeZeroVal(resultType));
  }
  const Effect e = effect;
  if (e == Effect::MallocHost || e == Effect::MallocDevice || e == Effect::MallocAsync || e == Effect::StreamCreate ||
      e == Effect::EventCreate || e == Effect::HostRegister)
    applyAllocation(call, c, s);
  else
    applyCall(call, c, s);
  return true;
}

void AsyncLifetimeChecker::checkLocation(SVal location, bool isLoad, const Stmt *statement, CheckerContext &c) const {
  auto *r = base(location);
  if (!r)
    return;
  uint64_t accessBits = 0;
  if (const auto *expr = dyn_cast_or_null<Expr>(statement))
    if (!expr->getType()->isIncompleteType() && !expr->getType()->isVoidType())
      accessBits = c.getASTContext().getTypeSize(expr->getType());
  hostAccess(c, c.getState(), r, accessBits, isLoad, "Host reads the destination before asynchronous D2H completion",
             "Host writes a buffer still used by an asynchronous copy");
}
void AsyncLifetimeChecker::checkBind(SVal location, SVal value, const Stmt *, bool, CheckerContext &c) const {
  auto *destination = base(location), *source = base(value);
  auto s = c.getState();
  if (!destination || !source || destination->hasMemorySpace<StackSpaceRegion>(s))
    return;
  for (auto op : s->get<PendingOps>())
    if (overlaps(s, source, op.region)) {
      cudalifetime::info().coverageUnknown.insert("pending region stored outside the local stack: subsequent accesses "
                                                  "through opaque ownership are not modeled");
      if (op.access != DeviceUse)
        report(c, s, Rule::CL1, "Host buffer escapes to nonlocal storage while asynchronous work remains pending",
               op.memory, "S2", &op, true);
    }
}
void AsyncLifetimeChecker::checkPostStmt(const CUDAKernelCallExpr *call, CheckerContext &c) const {
  ++cudalifetime::info().kernelLaunches;
  auto s = c.getState();
  const CallExpr *config = call->getConfig();
  Stream stream =
      config && config->getNumArgs() > 3 ? streamIdentity(s, kernelValue(config->getArg(3), c)) : defaultStream();
  if (!stream)
    ++cudalifetime::info().unmodeled["kernel-launch:unknown-stream"];
  s = enqueue(s, stream,
              config && config->getNumArgs() > 3 ? kernelValue(config->getArg(3), c)
                                                 : c.getSValBuilder().makeZeroVal(c.getASTContext().VoidPtrTy));
  for (const auto *argExpr : call->arguments()) {
    const MemRegion *r = base(kernelValue(argExpr, c));
    if (!r && argExpr->getType()->isPointerType())
      ++cudalifetime::info().unmodeled["kernel-launch:unknown-pointer-operand"];
    deviceUse(c, s, r, stream, "Kernel launch");
    s = pending(s, r, stream, DeviceUse, call, c);
  }
  c.addTransition(s);
}
void AsyncLifetimeChecker::checkEndFunction(const ReturnStmt *, CheckerContext &c) const {
  auto s = c.getState();
  const auto *frame = c.getStackFrame();
  for (auto op : s->get<PendingOps>()) {
    if (op.access == DeviceUse || !op.region->hasMemorySpace<StackSpaceRegion>(s))
      continue;
    if (auto *space = dyn_cast<StackSpaceRegion>(op.region->getMemorySpace(s)))
      if (space->getStackFrame() == frame)
        report(c, s, Rule::CL2, "Stack buffer lifetime ends while asynchronous work may still access it", op.memory,
               "S4", &op);
  }
}
} // namespace
bool cudalifetime::summarize(const FunctionDecl *decl, Model &out) {
  const auto &summary = AsyncLifetimeChecker::summaryOf(decl);
  if (!summary.copy && !summary.sync)
    return false;
  out = summary.copy ? *summary.copy : *summary.sync;
  return true;
}
bool cudalifetime::isCudaApi(const FunctionDecl *decl) {
  if (!decl || isa<CXXMethodDecl>(decl) || !decl->getIdentifier() || !decl->getName().starts_with("cu"))
    return false;
  static const std::set<std::string> headers = {"cuda.h",
                                                "cuda_runtime.h",
                                                "cuda_runtime_api.h",
                                                "cuda_device_runtime_api.h",
                                                "cudaProfiler.h",
                                                "cuda_profiler_api.h",
                                                "cudaGL.h",
                                                "cuda_gl_interop.h",
                                                "cudaEGL.h",
                                                "cuda_egl_interop.h",
                                                "cudaVDPAU.h",
                                                "cuda_vdpau_interop.h",
                                                "cudaD3D9.h",
                                                "cudaD3D10.h",
                                                "cudaD3D11.h",
                                                "cuda_d3d9_interop.h",
                                                "cuda_d3d10_interop.h",
                                                "cuda_d3d11_interop.h"};
  const SourceManager &sources = decl->getASTContext().getSourceManager();
  for (const FunctionDecl *redecl : decl->redecls()) {
    const FunctionDecl *pattern = redecl->getTemplateInstantiationPattern();
    const SourceLocation location = sources.getFileLoc((pattern ? pattern : redecl)->getLocation());
    if (location.isInvalid())
      continue;
    if (auto file = sources.getFileEntryRefForID(sources.getFileID(location)))
      if (headers.count(llvm::sys::path::filename(file->getName()).str()))
        return true;
  }
  return false;
}
void cudalifetime::registerChecker(CheckerRegistry &registry) {
  registry.addChecker<AsyncLifetimeChecker>("cuda.AsyncLifetime",
                                            "Checks CUDA host and stream-ordered asynchronous buffer lifetimes", "");
}
