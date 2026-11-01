#include "checker/AsyncLifetime.h"
#include "clang/AST/Attr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
using namespace clang;
using namespace ento;
namespace {
using Stream = uint64_t;
enum Memory { Unknown, Pinned, Pageable, Device, Ordered };
enum Access { HostRead, HostWrite, DeviceUse };
struct Pair {
  uint64_t first = 0, second = 0;
  bool operator==(const Pair &o) const { return first == o.first && second == o.second; }
  bool operator<(const Pair &o) const { return std::tie(first, second) < std::tie(o.first, o.second); }
  void Profile(llvm::FoldingSetNodeID &id) const { id.AddInteger(first); id.AddInteger(second); }
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
  bool operator==(const Pending &o) const { return std::tie(region, stream, seq, access, memory, origin, context, bits) == std::tie(o.region, o.stream, o.seq, o.access, o.memory, o.origin, o.context, o.bits); }
  bool operator<(const Pending &o) const { return std::tie(region, stream, seq, access, memory, origin, context, bits) < std::tie(o.region, o.stream, o.seq, o.access, o.memory, o.origin, o.context, o.bits); }
  void Profile(llvm::FoldingSetNodeID &id) const { id.AddPointer(region); id.AddInteger(stream); id.AddInteger(seq); id.AddInteger(access); id.AddInteger(memory); id.AddPointer(origin); id.AddPointer(context); id.AddInteger(bits); }
};
struct Allocation {
  Stream stream = 0;
  unsigned seq = 0;
  bool freed = false;
  Stream freeStream = 0;
  unsigned freeSeq = 0;
  const Stmt *origin = nullptr;
  const StackFrame *context = nullptr;
  bool operator==(const Allocation &o) const { return std::tie(stream, seq, freed, freeStream, freeSeq, origin, context) == std::tie(o.stream, o.seq, o.freed, o.freeStream, o.freeSeq, o.origin, o.context); }
  void Profile(llvm::FoldingSetNodeID &id) const { id.AddInteger(stream); id.AddInteger(seq); id.AddBoolean(freed); id.AddInteger(freeStream); id.AddInteger(freeSeq); id.AddPointer(origin); id.AddPointer(context); }
};
}
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
    static int tag; id.AddPointer(&tag); op.Profile(id);
  }
  PathDiagnosticPieceRef VisitNode(const ExplodedNode *node, BugReporterContext &context,
                                  PathSensitiveBugReport &) override {
    if (!op.origin || !op.context || node->pred_empty()) return nullptr;
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
    static int tag; id.AddPointer(&tag); id.AddPointer(region);
    allocation.Profile(id); id.AddBoolean(stillPending);
  }
  PathDiagnosticPieceRef VisitNode(const ExplodedNode *node, BugReporterContext &context,
                                  PathSensitiveBugReport &) override {
    if (!allocation.origin || !allocation.context || node->pred_empty()) return nullptr;
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
class AsyncLifetimeChecker : public Checker<check::PreCall, check::PostCall, eval::Call,
    check::Location, check::Bind, check::PostStmt<CUDAKernelCallExpr>, check::EndFunction,
    check::PointerEscape, check::EndAnalysis, check::LiveSymbols, check::BeginFunction> {
  // One bug type per (rule, level, memory kind, semantic ID), so reports that
  // differ only in those stay distinct, as CSA deduplicates by bug type.
  mutable std::map<std::string, std::unique_ptr<BugType>> bugTypes;
  static const MemRegion *base(SVal v) {
    auto *r = v.getAsRegion();
    return r ? r->StripCasts() : nullptr;
  }
  static bool sameAddress(ProgramStateRef s, const MemRegion *a, const MemRegion *b) {
    if (!a || !b) return false;
    if (a == b) return true;
    auto equal = s->getStateManager().getSValBuilder()
        .evalEQ(s, loc::MemRegionVal(a), loc::MemRegionVal(b)).getAs<DefinedOrUnknownSVal>();
    return equal && !s->assume(*equal, false);
  }
  static const MemRegion *allocationRegion(ProgramStateRef s, const MemRegion *r) {
    if (!r) return nullptr;
    auto *root = r->getBaseRegion();
    if (s->get<Allocations>(root)) return root;
    for (auto entry : s->get<Allocations>()) if (sameAddress(s, root, entry.first)) return entry.first;
    return nullptr;
  }
  static uint64_t identity(SVal v) {
    if (auto n = v.getAs<loc::ConcreteInt>()) return n->getValue()->getZExtValue() * 2 + 2;
    if (auto n = v.getAs<nonloc::ConcreteInt>()) return n->getValue()->getZExtValue() * 2 + 2;
    if (auto *r = base(v)) return reinterpret_cast<uintptr_t>(r) | 1;
    if (auto s = v.getAsSymbol()) return reinterpret_cast<uintptr_t>(s) | 1;
    return 0;
  }
  static bool sameValue(ProgramStateRef s, SVal a, SVal b) {
    // SValBuilder's pointer/integer comparison requires a cast first. Unknown
    // CUDA operands must not reach evalEQ as incompatible Loc/NonLoc values.
    if (a.isUnknownOrUndef() || b.isUnknownOrUndef() ||
        bool(a.getAs<Loc>()) != bool(b.getAs<Loc>())) return false;
    auto equal = s->getStateManager().getSValBuilder().evalEQ(s, a, b).getAs<DefinedOrUnknownSVal>();
    return equal && !s->assume(*equal, false);
  }
  static uint64_t streamIdentity(ProgramStateRef s, SVal value) {
    uint64_t raw = identity(value);
    auto &builder = s->getStateManager().getSValBuilder();
    if (raw == 2 || sameValue(s, value, builder.makeZeroVal(builder.getContext().VoidPtrTy)))
      return cudalifetime::info().perThread ? 6 : 4;
    for (auto entry : s->get<StreamValues>()) if (sameValue(s, entry.second, value)) return entry.first;
    return identity(value);
  }
  static uint64_t eventIdentity(ProgramStateRef s, SVal value) {
    for (auto entry : s->get<EventValues>()) if (sameValue(s, entry.second, value)) return entry.first;
    return identity(value);
  }
  static unsigned sequence(ProgramStateRef s, Stream stream) {
    auto *n = s->get<Sequences>(stream); return n ? *n : 0;
  }
  static unsigned completed(ProgramStateRef s, Stream stream) {
    auto *n = s->get<HostCompleted>(stream); return n ? *n : 0;
  }
  static Stream defaultStream() { return cudalifetime::info().perThread ? 6 : 4; }
  static bool capturing(ProgramStateRef s, Stream stream) { return stream && s->get<CaptureOrigins>(stream); }
  static void unknownDirection() {
    cudalifetime::info().coverageUnknown.insert("cudaMemcpyDefault copy with an operand of unknown memory kind: the copy direction for that operand is not modeled");
  }
  static void noteCapture() {
    cudalifetime::info().coverageUnknown.insert("work enqueued during stream capture is recorded into a graph, not executed; graph contents and launches are not modeled");
  }
  static Memory memory(ProgramStateRef s, const MemRegion *r, uint64_t bits = 0) {
    if (!r) return Unknown;
    for (auto entry : s->get<Memories>()) {
      if (const auto *extent = s->get<RegisteredExtents>(entry.first)) {
        auto ro = r->getAsOffset(), eo = entry.first->getAsOffset();
        if (!ro.isValid() || !eo.isValid() || ro.hasSymbolicOffset() || eo.hasSymbolicOffset() ||
            !sameAddress(s, ro.getRegion(), eo.getRegion()) || !bits || !*extent) continue;
        __int128 start = ro.getOffset(), registeredStart = eo.getOffset();
        if (start >= registeredStart && start + bits <= registeredStart + *extent) return Pinned;
      } else if (r == entry.first || r->isSubRegionOf(entry.first) || sameAddress(s, r, entry.first))
        return static_cast<Memory>(entry.second);
    }
    if (r->hasMemorySpace<StackSpaceRegion>(s)) return Pageable;
    return Unknown;
  }
  static unsigned kind(ProgramStateRef s, Stream stream) {
    // 1 legacy, 2 blocking, 3 nonblocking, 4 per-thread, 0 unknown.
    if (stream == 2) return cudalifetime::info().perThread ? 4 : 1;
    if (stream == 4) return 1; // cudaStreamLegacy == (cudaStream_t)1.
    if (stream == 6) return 4; // cudaStreamPerThread == (cudaStream_t)2.
    auto *v = s->get<StreamKinds>(stream); return v ? *v : 0;
  }
  static bool distinct(ProgramStateRef s, Stream a, Stream b) {
    if (!a || !b || a == b) return false;
    const bool aCreated = s->get<StreamKinds>(a), bCreated = s->get<StreamKinds>(b);
    const bool aReserved = a == 4 || a == 6, bReserved = b == 4 || b == 6;
    if ((aCreated && (bCreated || bReserved)) || (bCreated && aReserved) ||
        (aReserved && bReserved)) return true;
    const auto *av = s->get<StreamValues>(a), *bv = s->get<StreamValues>(b);
    if (!av || !bv) return false;
    if (av->isUnknownOrUndef() || bv->isUnknownOrUndef() ||
        bool(av->getAs<Loc>()) != bool(bv->getAs<Loc>())) return false;
    auto equal = s->getStateManager().getSValBuilder().evalEQ(s, *av, *bv).getAs<DefinedOrUnknownSVal>();
    return equal && !s->assume(*equal, true);

  }
  static bool ordered(ProgramStateRef s, Stream waiter, Stream source, unsigned seq) {
    if (completed(s, source) >= seq) return true;
    if (!waiter || !source) return false;
    if (waiter == source) return true;
    const auto *wv = s->get<StreamValues>(waiter), *sv = s->get<StreamValues>(source);
    if (wv && sv && sameValue(s, *wv, *sv)) return true;
    auto *bound = s->get<WaitBounds>({waiter, source});
    return bound && *bound >= seq;
  }
  static ProgramStateRef bound(ProgramStateRef s, Stream waiter, Stream source, unsigned seq) {
    auto *old = s->get<WaitBounds>({waiter, source});
    if (!old || *old < seq) s = s->set<WaitBounds>({waiter, source}, seq);
    return s;
  }
  static ProgramStateRef enqueue(ProgramStateRef s, Stream stream, SVal value = UnknownVal()) {
    if (!value.isUnknownOrUndef()) s = s->set<StreamValues>(stream, value);
    // Legacy's device ordering is independent of host completion.
    unsigned k = kind(s, stream);
    for (auto it : s->get<Sequences>()) {
      unsigned other = kind(s, it.first);
      // CUDA's per-thread default stream is blocking with respect to an
      // explicitly used legacy stream, but does not order ordinary streams.
      // CUDA 13.2.2 archive: stream-sync-behavior#per-thread-default-stream.
      if ((k == 1 && (other == 2 || other == 4)) ||
          ((k == 2 || k == 4) && other == 1)) {
        s = bound(s, stream, it.first, it.second);
        for (auto wait : s->get<WaitBounds>())
          if (wait.first.first == it.first)
            s = bound(s, stream, wait.first.second, wait.second);
      }
    }
    return s->set<Sequences>(stream, sequence(s, stream) + 1);
  }
  static ProgramStateRef retire(ProgramStateRef s, Stream stream, unsigned seq) {
    if (seq > completed(s, stream)) s = s->set<HostCompleted>(stream, seq);
    for (auto op : s->get<PendingOps>())
      if (op.stream == stream && op.seq <= seq) s = s->remove<PendingOps>(op);
    return s;
  }
  static ProgramStateRef syncStream(ProgramStateRef s, Stream stream) {
    if (!stream) return s;
    s = retire(s, stream, sequence(s, stream));
    for (auto wait : s->get<WaitBounds>())
      if (wait.first.first == stream) s = retire(s, wait.first.second, wait.second);
    return s;
  }
  static bool overlaps(ProgramStateRef s, const MemRegion *a, const MemRegion *b) {
    return a && b && (a == b || a->isSubRegionOf(b) || b->isSubRegionOf(a) || sameAddress(s, a, b));
  }
  static unsigned rangeOverlap(ProgramStateRef s, const MemRegion *a, uint64_t aBits,
                               const MemRegion *b, uint64_t bBits) {
    auto ao = a->getAsOffset(), bo = b->getAsOffset();
    if (ao.isValid() && bo.isValid() && sameAddress(s, ao.getRegion(), bo.getRegion())) {
      if (ao.hasSymbolicOffset() || bo.hasSymbolicOffset() || !aBits || !bBits) return 1;
      __int128 aStart = ao.getOffset(), bStart = bo.getOffset();
      return aStart < bStart + bBits && bStart < aStart + aBits ? 2 : 0;
    }
    return overlaps(s, a, b) ? 1 : 0;
  }
  void report(CheckerContext &c, ProgramStateRef s, unsigned rule,
              const std::string &description, Memory mem, const char *semantic,
              const Pending *op = nullptr, bool contract = false, bool informational = false,
              const Allocation *allocation = nullptr, const MemRegion *allocationKey = nullptr) const {
    auto *node = c.generateNonFatalErrorNode(s);
    if (!node) return;
    const char *names[] = {"unknown", "pinned", "pageable", "device", "stream-ordered"};
    const bool certain = !contract && (mem == Pinned || mem == Ordered);
    const cudalifetime::ReportKind kind{"CL" + std::to_string(rule), informational ? "note" : certain ? "error" : "warning",
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
        r->addNote("Asynchronous operation is queued here; completion has not been established", PathDiagnosticLocation::createBegin(op->origin, c.getSourceManager(), op->context));
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
    if (!value.isUnknownOrUndef()) return value;
    QualType originalType = expr->getType();
    expr = expr->IgnoreParenCasts();
    if (const auto *arg = dyn_cast<CXXDefaultArgExpr>(expr)) value = kernelValue(arg->getExpr(), c);
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
  void deviceUse(CheckerContext &c, ProgramStateRef s, const MemRegion *r,
                 Stream stream, const char *action) const {
    if (!r || capturing(s, stream)) return;
    if (auto *key = allocationRegion(s, r)) {
      const auto *alloc = s->get<Allocations>(key);
      if (alloc->freed) report(c, s, 3, std::string(action) + " uses an allocation after its asynchronous free was issued", Ordered, "S6", nullptr, false, false, alloc, key);
      else if (!ordered(s, stream, alloc->stream, alloc->seq))
        report(c, s, 3, std::string(action) + " uses a stream-ordered allocation without an allocation dependency", Ordered, "S5", nullptr, !distinct(s, stream, alloc->stream), false, alloc, key);
    }
  }
  static std::string name(const CallEvent &call) { return declName(dyn_cast_or_null<FunctionDecl>(call.getDecl())); }
  static std::string declName(const FunctionDecl *fd) {
    if (fd) {
      auto qualified = fd->getQualifiedNameAsString();
      if (cudalifetime::info().models.count(qualified)) return qualified;
      auto n = fd->getNameAsString();
      for (const char *suffix : {"_ptsz", "_ptds"})
        if (n.size() > 5 && n.compare(n.size() - 5, 5, suffix) == 0) n.resize(n.size() - 5);
      return n;
    }
    return {};
  }
  // Device, version and occupancy queries and settings: no stream work, no user buffer.
  static bool deviceQuery(const std::string &n) {
    static const std::set<std::string> names = {
      "cudaGetDevice", "cudaSetDevice", "cudaGetDeviceCount", "cudaGetDeviceProperties", "cudaDeviceGetAttribute",
      "cudaDeviceCanAccessPeer", "cudaDriverGetVersion", "cudaRuntimeGetVersion", "cudaMemGetInfo",
      "cudaDeviceSetLimit", "cudaDeviceGetLimit", "cudaFuncSetAttribute", "cudaFuncGetAttributes",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor", "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      "cudaCreateChannelDesc", "cudaStreamGetId", "cudaStreamGetPriority", "cudaDeviceGetStreamPriorityRange",
      "cudaGetDeviceFlags", "cudaSetDeviceFlags", "cudaChooseDevice", "cudaProfilerStart", "cudaProfilerStop"};
    return names.count(n) != 0;
  }
  static cudalifetime::Model model(const std::string &n) {
    if (auto it = cudalifetime::info().models.find(n); it != cudalifetime::info().models.end()) return it->second;
    cudalifetime::Model m; m.function = n;
    if (n == "cudaMemcpyAsync") { m.effect = "enqueue_copy"; m.args = {{"dst",0},{"src",1},{"size",2},{"kind",3},{"stream",4}}; }
    else if (n == "cudaMemcpyBatchAsync") {
      // Arrays of operands; the direction follows from the pointers, as with
      // cudaMemcpyDefault. Only the first entry is modeled.
      m.effect = "enqueue_copy"; m.args = {{"dst",0},{"src",1},{"size",2},{"count",3},{"stream",7}};
      m.constants = {{"kind", 4}}; m.elements = {"dst", "src", "size"};
    }
    else if (n == "cudaMemcpy2DAsync") { m.effect = "enqueue_copy"; m.args = {{"dst",0},{"src",2},{"kind",6},{"stream",7}}; }
    else if (n == "cudaMemcpy") { m.effect = "copy"; m.args = {{"dst",0},{"src",1},{"size",2},{"kind",3}}; }
    else if (n == "cudaMemcpy2D") { m.effect = "copy"; m.args = {{"dst",0},{"src",2},{"kind",6}}; }
    else if (n == "cudaMemset") { m.effect = "memset"; m.args = {{"ptr",0},{"size",2}}; }
    else if (n == "cudaMemsetAsync") { m.effect = "memset_async"; m.args = {{"ptr",0},{"size",2},{"stream",3}}; }
    else if (n == "cudaStreamBeginCapture" || n == "cudaStreamBeginCaptureToGraph") { m.effect = "capture_begin"; m.args = {{"stream",0}}; }
    else if (n == "cudaStreamEndCapture") { m.effect = "capture_end"; m.args = {{"stream",0},{"ptr",1}}; }
    else if (n == "cudaGraphLaunch") { m.effect = "graph_launch"; m.args = {{"stream",1}}; }
    else if (n == "cudaPointerGetAttributes" || n == "cudaHostGetDevicePointer" || n == "cudaHostGetFlags") m.effect = "address_only";
    else if (n == "cudaMallocHost" || n == "cudaHostAlloc") { m.effect = "malloc_host"; m.args = {{"ptr",0}}; }
    else if (n == "cudaHostRegister") { m.effect = "host_register"; m.args = {{"ptr",0},{"size",1}}; }
    else if (n == "cudaHostUnregister") { m.effect = "host_unregister"; m.args = {{"ptr",0}}; }
    else if (n == "cudaMalloc") { m.effect = "malloc_device"; m.args = {{"ptr",0}}; }
    else if (n == "cudaMallocAsync" || n == "cudaMallocFromPoolAsync") { m.effect = "malloc_async"; m.args = {{"ptr",0},{"stream",n == "cudaMallocAsync" ? 2u : 3u}}; }
    else if (n == "cudaFreeAsync") { m.effect = "free_async"; m.args = {{"ptr",0},{"stream",1}}; }
    else if (n == "cudaFreeHost") { m.effect = "free_host"; m.args = {{"ptr",0}}; }
    else if (n == "cudaFree") { m.effect = "free_device"; m.args = {{"ptr",0}}; }
    else if (n == "cudaDeviceSynchronize") m.effect = "sync_device";
    else if (n == "cudaStreamSynchronize" || n == "cudaStreamQuery") { m.effect = "sync_stream"; m.args = {{"stream",0}}; }
    else if (n == "cudaEventSynchronize" || n == "cudaEventQuery") { m.effect = "sync_event"; m.args = {{"event",0}}; }
    else if (n == "cudaEventElapsedTime") { m.effect = "event_elapsed"; m.args = {{"ms",0},{"start",1},{"end",2}}; }
    else if (n == "cudaEventRecord" || n == "cudaEventRecordWithFlags") { m.effect = "event_record"; m.args = {{"event",0},{"stream",1}}; }
    else if (n == "cudaStreamWaitEvent") { m.effect = "stream_wait_event"; m.args = {{"stream",0},{"event",1}}; }
    else if (n == "cudaStreamCreate" || n == "cudaStreamCreateWithFlags" || n == "cudaStreamCreateWithPriority") { m.effect = "stream_create"; m.args = {{"ptr",0},{"flags",1}}; }
    else if (n == "cudaEventCreate" || n == "cudaEventCreateWithFlags") { m.effect = "event_create"; m.args = {{"ptr",0}}; if (n == "cudaEventCreateWithFlags") m.args["flags"] = 1; }
    else if (n == "cudaStreamDestroy" || n == "cudaEventDestroy") m.effect = "destroy";
    else if (n == "cudaGetLastError" || n == "cudaPeekAtLastError" || n == "cudaGetErrorString" || n == "cudaGetErrorName" ||
             n == "cudaGraphInstantiate" || n == "cudaGraphInstantiateWithFlags" || n == "cudaGraphDestroy" || n == "cudaGraphExecDestroy" ||
             deviceQuery(n)) m.effect = "observe";
    return m;
  }
public:
  // Summaries of bodies: the copy a function queues on its own parameters,
  // composed from the CUDA calls, models and other bodies it calls. They are
  // flow-insensitive "may" summaries: a copy on any path counts, and a body
  // that synchronizes anything gets none, since it may complete its copies.
  // A wrapper whose only data member is a handle (cuda_stream_view, stream_ref).
  static bool handleRecord(QualType type) {
    if (!type.isNull() && type->isPointerType()) type = type->getPointeeType();
    const auto *record = type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl();
    if (!record || !(record = record->getDefinition()) || record->getNumBases() || record->isPolymorphic()) return false;
    return std::distance(record->field_begin(), record->field_end()) == 1;
  }
  struct Summary { bool done = false, synchronizes = false, copies = false; std::optional<cudalifetime::Model> copy, sync; };
  struct Forwarded {
    // level -1: the address of the parameter; 0: its value; 1: an element of
    // the array it points to.
    int param = -1, level = 0; bool constant = false, receiver = false; int64_t value = 0;
    bool valid() const { return param >= 0 || constant || receiver; }
  };
  // Which parameter of fd an argument expression passes on, or its constant
  // value. Buffers forward as a pointer parameter or the address of a
  // reference parameter; a handle may also be read through a wrapper's
  // accessor or conversion (stream.value(), a cuda_stream_view to stream_ref).
  static Forwarded forwarded(const Expr *e, const FunctionDecl *fd, bool handle) {
    Forwarded out;
    for (unsigned depth = 0; e && depth < 16; ++depth) {
      e = e->IgnoreImplicit()->IgnoreParens();
      if (const auto *cast = dyn_cast<ExplicitCastExpr>(e)) { e = cast->getSubExpr(); continue; }
      if (const auto *temp = dyn_cast<MaterializeTemporaryExpr>(e)) { e = temp->getSubExpr(); continue; }
      if (handle) {
        if (const auto *construct = dyn_cast<CXXConstructExpr>(e); construct && construct->getNumArgs() == 1) { e = construct->getArg(0); continue; }
        if (const auto *member = dyn_cast<CXXMemberCallExpr>(e); member && !member->getNumArgs() && member->getImplicitObjectArgument() &&
            handleRecord(member->getImplicitObjectArgument()->getType())) {
          e = member->getImplicitObjectArgument(); continue;
        }
      }
      break;
    }
    if (!e) return out;
    // A handle the receiver holds: this->stream_, or value() on this.
    if (handle) {
      const Expr *object = e;
      if (const auto *field = dyn_cast<MemberExpr>(e)) object = field->getBase()->IgnoreParenImpCasts();
      if (isa<CXXThisExpr>(object)) { out.receiver = handleRecord(object->getType()); return out; }
    }
    auto index = [&](const ValueDecl *decl) -> int {
      const auto *param = dyn_cast_or_null<ParmVarDecl>(decl);
      if (!param) return -1;
      for (unsigned i = 0; i < fd->getNumParams(); ++i) if (fd->getParamDecl(i) == param) return static_cast<int>(i);
      return -1;
    };
    if (const auto *unary = dyn_cast<UnaryOperator>(e); unary && unary->getOpcode() == UO_AddrOf) {
      // &value of a reference parameter is the address the caller passed; &p
      // of any other parameter is the address of the callee's copy of it.
      if (const auto *ref = dyn_cast<DeclRefExpr>(unary->getSubExpr()->IgnoreParens())) {
        out.param = index(ref->getDecl());
        if (!ref->getDecl()->getType()->isReferenceType()) out.level = -1;
      }
      return out;
    }
    const Expr *array = nullptr;
    if (const auto *subscript = dyn_cast<ArraySubscriptExpr>(e)) array = subscript->getBase();
    else if (const auto *unary = dyn_cast<UnaryOperator>(e); unary && unary->getOpcode() == UO_Deref) array = unary->getSubExpr();
    if (array) {
      if (const auto *ref = dyn_cast<DeclRefExpr>(array->IgnoreParenImpCasts()); ref && ref->getDecl()->getType()->isPointerType()) {
        out.param = index(ref->getDecl());
        out.level = 1;
      }
      return out;
    }
    if (const auto *ref = dyn_cast<DeclRefExpr>(e)) {
      QualType type = ref->getDecl()->getType().getNonReferenceType();
      if (type->isPointerType() || type->isIntegralOrEnumerationType() || (handle && type->getAsCXXRecordDecl()))
        out.param = index(ref->getDecl());
      if (out.param >= 0) return out;
    }
    Expr::EvalResult result;
    if (!e->isValueDependent() && !e->isTypeDependent() && e->EvaluateAsInt(result, fd->getASTContext())) {
      out.constant = true; out.value = result.Val.getInt().getExtValue();
    }
    return out;
  }
  // Declared noreturn, or every path through the body ends in a throw or a
  // noreturn call (a syntactic check: statements in sequence, if/else).
  static bool endsInThrow(const Stmt *statement, unsigned depth = 0) {
    if (!statement || depth > 16) return false;
    if (const auto *expr = dyn_cast<Expr>(statement)) {
      expr = expr->IgnoreImplicit()->IgnoreParens();
      if (isa<CXXThrowExpr>(expr)) return true;
      if (const auto *call = dyn_cast<CallExpr>(expr)) return call->getDirectCallee() && call->getDirectCallee()->isNoReturn();
      return false;
    }
    if (const auto *block = dyn_cast<CompoundStmt>(statement)) {
      for (const auto *child : block->body()) if (endsInThrow(child, depth + 1)) return true;
      return false;
    }
    if (const auto *branch = dyn_cast<IfStmt>(statement))
      return endsInThrow(branch->getThen(), depth + 1) && endsInThrow(branch->getElse(), depth + 1);
    return false;
  }
  // Whether every path from the body's entry to its normal exit passes the
  // call: paths that throw or call a function that never returns don't count.
  static bool alwaysReached(const FunctionDecl *fd, const CallExpr *call) {
    auto cfg = CFG::buildCFG(fd, fd->getBody(), &fd->getASTContext(), CFG::BuildOptions());
    if (!cfg) return false;
    llvm::SmallPtrSet<const CFGBlock *, 4> at;
    auto deadEnd = [](const CFGBlock *block) {
      if (block->hasNoReturnElement() || isa_and_nonnull<CXXThrowExpr>(block->getTerminatorStmt())) return true;
      for (const auto &element : *block)
        if (auto statement = element.getAs<CFGStmt>())
          if (const auto *inner = dyn_cast<CallExpr>(statement->getStmt()); inner && inner->getDirectCallee() && neverReturns(inner->getDirectCallee()))
            return true;
      return false;
    };
    for (const CFGBlock *block : *cfg)
      for (const auto &element : *block)
        if (auto statement = element.getAs<CFGStmt>(); statement && statement->getStmt() == call) at.insert(block);
    if (at.empty()) return false;
    std::vector<const CFGBlock *> work{&cfg->getEntry()};
    llvm::SmallPtrSet<const CFGBlock *, 32> seen;
    while (!work.empty()) {
      const CFGBlock *block = work.back(); work.pop_back();
      if (!block || !seen.insert(block).second || at.count(block)) continue;
      if (block == &cfg->getExit()) return false;
      if (deadEnd(block)) continue;
      for (const auto &next : block->succs()) work.push_back(next.getReachableBlock());
    }
    return true;
  }
  static bool neverReturns(const FunctionDecl *fd) {
    if (fd->isNoReturn()) return true;
    const FunctionDecl *definition = fd->getDefinition();
    return definition && definition->hasBody() && endsInThrow(definition->getBody());
  }
  static const Summary &summaryOf(const FunctionDecl *fd) {
    static std::map<const FunctionDecl *, Summary> memo;
    fd = fd->getCanonicalDecl();
    auto &entry = memo[fd];
    if (entry.done) return entry;
    entry.done = true;  // a recursive call sees an empty summary
    const FunctionDecl *definition = fd->getDefinition();
    if (!definition || !definition->hasBody() || definition->isDependentContext()) return entry;
    struct Calls : RecursiveASTVisitor<Calls> {
      std::vector<const CallExpr *> calls;
      bool VisitCallExpr(CallExpr *call) { calls.push_back(call); return true; }
      bool TraverseLambdaExpr(LambdaExpr *) { return true; }  // runs only if called
    } visitor;
    visitor.TraverseStmt(definition->getBody());
    Summary result; result.done = true;
    std::vector<cudalifetime::Model> copies;
    for (const CallExpr *call : visitor.calls) {
      const FunctionDecl *callee = call->getDirectCallee();
      // A callee that never returns (an error handler that synchronizes, then
      // throws) ends its path: what it does is never followed by the caller's code.
      if (!callee || isa<CXXOperatorCallExpr>(call) || neverReturns(callee)) continue;
      cudalifetime::Model m = model(declName(callee));
      if (m.effect.empty() && !cudalifetime::isCudaApi(callee)) {
        const Summary &inner = summaryOf(callee);
        result.synchronizes |= inner.synchronizes;
        if (inner.copy) m = *inner.copy;
        else if (inner.sync) m = *inner.sync;
      }
      const auto &e = m.effect;
      if (e == "sync_stream" || e == "sync_device" || e == "sync_event" || e == "event_elapsed" || e == "copy")
        result.synchronizes = true;
      // A synchronization on every path through the body (a "must" claim,
      // unlike copies) makes a summary of its own.
      if ((e == "sync_stream" || e == "sync_device") && !result.sync && !neverReturns(definition) && alwaysReached(definition, call)) {
        cudalifetime::Model sync; sync.function = definition->getQualifiedNameAsString(); sync.effect = e; sync.assumption = true;
        Forwarded f;
        if (e == "sync_stream") {
          if (auto constant = m.constants.find("stream"); constant != m.constants.end()) { f.constant = true; f.value = constant->second; }
          else if (m.receiver.count("stream")) {
            if (const auto *member = dyn_cast<CXXMemberCallExpr>(call); member && member->getImplicitObjectArgument())
              f = forwarded(member->getImplicitObjectArgument(), definition, true);
          } else if (auto it = m.args.find("stream"); it != m.args.end() && it->second < call->getNumArgs())
            f = forwarded(call->getArg(it->second), definition, true);
          if (f.param >= 0 && f.level == 0) sync.args["stream"] = static_cast<unsigned>(f.param);
          else if (f.receiver) sync.receiver.insert("stream");
          else if (f.constant) sync.constants["stream"] = f.value;
        }
        if (e == "sync_device" || f.valid()) result.sync = sync;
      }
      if (e != "enqueue_copy") continue;
      cudalifetime::Model copy; copy.function = definition->getQualifiedNameAsString(); copy.effect = e; copy.assumption = true;
      bool forwardsBuffer = false, forwardsStream = false;
      for (const char *key : {"dst", "src", "size", "kind", "stream", "count"}) {
        Forwarded f;
        if (auto constant = m.constants.find(key); constant != m.constants.end()) { f.constant = true; f.value = constant->second; }
        else if (m.receiver.count(key)) {
          if (const auto *member = dyn_cast<CXXMemberCallExpr>(call); member && member->getImplicitObjectArgument())
            f = forwarded(member->getImplicitObjectArgument(), definition, true);
        } else if (auto it = m.args.find(key); it != m.args.end() && it->second < call->getNumArgs()) {
          f = forwarded(call->getArg(it->second), definition, std::string(key) == "stream");
          if (m.elements.count(key)) {
            // The callee's operand is the first element of this argument.
            if (f.constant) f = Forwarded();
            ++f.level;
          }
        }
        if (f.param >= 0 && (f.level < 0 || f.level > 1)) f = Forwarded();
        if (f.param >= 0 && f.level == 1) copy.elements.insert(key);
        if (f.param >= 0) copy.args[key] = static_cast<unsigned>(f.param);
        else if (f.receiver && std::string(key) == "stream") copy.receiver.insert(key);
        else if (f.constant && (std::string(key) == "kind" || std::string(key) == "stream")) copy.constants[key] = f.value;
        else if (f.constant && std::string(key) == "size" && f.value >= 0) copy.constants[key] = f.value;
        forwardsBuffer |= (std::string(key) == "dst" || std::string(key) == "src") && f.param >= 0;
        forwardsStream |= std::string(key) == "stream" && f.valid();
      }
      // A count matters only while an operand is still an array.
      if (copy.elements.empty()) { copy.args.erase("count"); copy.constants.erase("count"); }
      if (forwardsBuffer && forwardsStream) copies.push_back(copy);
    }
    result.copies = !copies.empty();
    bool agree = !copies.empty();
    auto withoutCount = [](cudalifetime::Model m) { m.args.erase("count"); m.constants.erase("count"); return m; };
    for (const auto &copy : copies) {
      auto a = withoutCount(copy), b = withoutCount(copies[0]);
      agree &= a.args == b.args && a.constants == b.constants && a.elements == b.elements && a.receiver == b.receiver;
    }
    if (agree && !result.synchronizes) {
      result.copy = copies[0];
      for (const auto &copy : copies) if (copy.args.count("count")) result.copy->args["count"] = copy.args.at("count");
    }
    memo[fd] = result;
    return memo[fd];
  }
  static SVal arg(ProgramStateRef s, const CallEvent &call, const cudalifetime::Model &m, const char *key) {
    auto &manager = s->getStateManager();
    if (auto constant = m.constants.find(key); constant != m.constants.end())
      return manager.getSValBuilder().makeIntVal(constant->second, manager.getContext().IntTy);
    if (m.receiver.count(key)) {
      // The operand is the receiver object, such as a stream wrapper's synchronize().
      const auto *instance = dyn_cast<CXXInstanceCall>(&call);
      const auto *method = instance ? dyn_cast_or_null<CXXMethodDecl>(instance->getDecl()) : nullptr;
      if (!method) return UnknownVal();
      SVal object = instance->getCXXThisVal();
      return handle(s, object, method->getThisType()->getPointeeType());
    }
    auto it = m.args.find(key);
    if (it == m.args.end() || it->second >= call.getNumArgs()) return UnknownVal();
    SVal value = call.getArgSVal(it->second);
    const std::string k = key;
    if (m.elements.count(k)) {
      // An array of operands: the first entry. A count other than one leaves
      // the remaining entries unmodeled.
      auto count = m.args.count("count") && m.args.at("count") < call.getNumArgs() ? call.getArgSVal(m.args.at("count")).getAs<nonloc::ConcreteInt>() : std::nullopt;
      if (!count || count->getValue()->getZExtValue() != 1)
        cudalifetime::info().coverageUnknown.insert("batched copy with a count other than one: entries after the first are not modeled");
      const auto *pointee = value.getAsRegion();
      const auto *type = call.getArgExpr(it->second) ? call.getArgExpr(it->second)->getType()->getAs<PointerType>() : nullptr;
      if (!pointee || !type) return UnknownVal();
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
      if (!record) return value;
      record = record->getDefinition();
      if (!record || record->getNumBases() || record->isPolymorphic()) return UnknownVal();
      const FieldDecl *only = nullptr;
      unsigned fields = 0;
      for (const auto *field : record->fields()) { only = field; ++fields; }
      if (fields != 1) return UnknownVal();
      if (auto lazy = value.getAs<nonloc::LazyCompoundVal>()) {
        const auto *field = manager.getRegionManager().getFieldRegion(only, lazy->getRegion());
        value = manager.getStoreManager().getBinding(lazy->getStore(), loc::MemRegionVal(field), only->getType());
      } else if (const auto *object = dyn_cast_or_null<SubRegion>(value.getAsRegion())) {
        value = s->getSVal(manager.getRegionManager().getFieldRegion(only, object), only->getType());
      } else return UnknownVal();
      type = only->getType();
    }
    return UnknownVal();
  }
  static ProgramStateRef pending(ProgramStateRef s, const MemRegion *r, Stream stream,
                                  Access access, const Stmt *origin, CheckerContext &c, uint64_t bits = 0) {
    if (!r) return s;
    if (capturing(s, stream)) { noteCapture(); return s; }
    return s->add<PendingOps>({r, stream, sequence(s, stream), access, memory(s, r, bits), origin, c.getStackFrame(), bits});
  }
public:
  void checkLiveSymbols(ProgramStateRef s, SymbolReaper &reaper) const {
    for (auto entry : s->get<StreamValues>()) {
      if (auto symbol = entry.second.getAsSymbol()) reaper.markLive(symbol);
      if (auto *region = entry.second.getAsRegion()) reaper.markLive(region);
    }
    for (auto entry : s->get<EventValues>())
      if (auto symbol = entry.second.getAsSymbol()) reaper.markLive(symbol);
    for (auto region : s->get<PreservedVariables>()) {
      reaper.markLive(region);
      if (auto symbol = s->getSVal(region).getAsSymbol()) reaper.markLive(symbol);
    }
    for (auto entry : s->get<Memories>()) reaper.markLive(entry.first);
    for (auto entry : s->get<ContainerOwners>()) { reaper.markLive(entry.first); reaper.markLive(entry.second); }
    for (auto entry : s->get<ViewResults>()) reaper.markLive(entry.second);
    for (auto entry : s->get<Allocations>()) reaper.markLive(entry.first);
    for (auto entry : s->get<PendingOps>()) reaper.markLive(entry.region);
  }
  void checkBeginFunction(CheckerContext &c) const {
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
    if (!decl || !decl->hasBody()) return;
    references.TraverseStmt(decl->getBody());
    auto s = c.getState();
    for (auto *variable : references.variables)
      s = s->add<PreservedVariables>(s->getLValue(variable, c.getStackFrame()).getAsRegion());
    c.addTransition(s);
  }
  void checkEndAnalysis(ExplodedGraph &, BugReporter &, ExprEngine &engine) const {
    ++cudalifetime::info().functionsAnalyzed;
    if (engine.hasWorkRemaining()) {
      const Decl *decl = engine.getRootStackFrame()->getDecl();
      std::string function = isa<NamedDecl>(decl) ? cast<NamedDecl>(decl)->getQualifiedNameAsString() : "<anonymous>";
      cudalifetime::info().budgetExhausted.push_back(function);
    }
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
    if (conservative) s = opaqueCall(call, c, s);
    s = clearReach(s);
    if (n == "malloc" || n == "calloc" || n == "realloc")
      if (auto *region = base(call.getReturnValue())) s = s->set<Memories>(region, Pageable);
    s = trackContainers(call, s);
    // A body CSA declined to inline (a container method, or over an inlining
    // limit) still queues the copies its summary records.
    if (conservative) if (const auto *fd = dyn_cast_or_null<FunctionDecl>(call.getDecl()))
      if (fd->hasBody() && !cudalifetime::isCudaApi(fd))
        if (const auto &summary = summaryOf(fd); summary.copy || summary.sync) {
          const auto &applied = summary.copy ? *summary.copy : *summary.sync;
          cudalifetime::info().assumedModels.insert("generated summary: " + applied.function);
          applyCall(call, c, s, &applied);
          return;
        }
    if (s != c.getState()) c.addTransition(s);
  }
  // Host containers own their buffer: a pointer data() returns is host memory
  // that the container's destructor releases. A view (span, string_view)
  // constructed from a container, or from another view, refers to the same
  // owner. Owners are std::vector, std::basic_string and thrust::host_vector,
  // and classes derived from them.
  static bool named(const CXXRecordDecl *record, const std::set<std::string> &names, unsigned depth = 0) {
    if (!record || depth > 8 || !(record = record->getDefinition())) return false;
    if (names.count(record->getQualifiedNameAsString())) return true;
    for (const auto &b : record->bases()) if (named(b.getType()->getAsCXXRecordDecl(), names, depth + 1)) return true;
    return false;
  }
  static bool hostContainer(QualType type) {
    static const std::set<std::string> names = {"std::vector", "std::basic_string", "std::__cxx11::basic_string", "thrust::host_vector"};
    return named(type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl(), names);
  }
  static bool hostView(QualType type) {
    static const std::set<std::string> names = {"std::span", "std::basic_string_view", "cudf::host_span"};
    return named(type.isNull() ? nullptr : type.getNonReferenceType()->getAsCXXRecordDecl(), names);
  }
  static const MemRegion *owner(ProgramStateRef s, SVal value, QualType type) {
    const MemRegion *object = value.getAsRegion();
    if (!object) if (auto lazy = value.getAs<nonloc::LazyCompoundVal>()) object = lazy->getRegion();
    if (!object) return nullptr;
    object = object->StripCasts();  // a base-class subobject is the object itself
    if (hostContainer(type)) return object;
    if (!hostView(type)) return nullptr;
    if (const auto *known = s->get<ContainerOwners>(object)) return *known;
    // A view copied without a constructor call (a trivial copy into a
    // parameter) is bound to the original's value.
    if (const auto *typed = dyn_cast<TypedValueRegion>(object))
      if (auto lazy = s->getSVal(typed).getAs<nonloc::LazyCompoundVal>())
        if (const auto *known = s->get<ContainerOwners>(lazy->getRegion()->StripCasts())) return *known;
    // A parameter initialized by such a call: look the call up by its argument.
    if (const auto *param = dyn_cast<ParamVarRegion>(object))
      if (const auto *origin = dyn_cast_or_null<CallExpr>(param->getOriginExpr()); origin && !isa<CXXOperatorCallExpr>(origin) && param->getIndex() < origin->getNumArgs())
        if (const auto *known = s->get<ViewResults>(origin->getArg(param->getIndex())->IgnoreImplicit())) return *known;
    return nullptr;
  }
  // Moving a container moves its buffer: the moved-to object owns it, so the
  // moved-from one releases nothing (a container returned by value, or
  // assigned from std::move).
  static ProgramStateRef transfer(ProgramStateRef s, SVal from, SVal to) {
    const auto *source = from.getAsRegion(), *target = to.getAsRegion();
    if (!source || !target) return s;
    source = source->StripCasts(); target = target->StripCasts();
    for (auto entry : s->get<ContainerOwners>()) if (entry.second == source) s = s->set<ContainerOwners>(entry.first, target);
    return s;
  }
  static ProgramStateRef trackContainers(const CallEvent &call, ProgramStateRef s) {
    if (const auto *ctor = dyn_cast<CXXConstructorCall>(&call); ctor && ctor->getDecl()->isMoveConstructor() &&
        call.getNumArgs() == 1 && hostContainer(ctor->getDecl()->getThisType()->getPointeeType()))
      return transfer(s, call.getArgSVal(0), ctor->getCXXThisVal());
    if (const auto *assign = dyn_cast<CXXMemberOperatorCall>(&call))
      if (const auto *method = dyn_cast_or_null<CXXMethodDecl>(assign->getDecl()); method && method->isMoveAssignmentOperator() &&
          call.getNumArgs() == 1 && hostContainer(method->getThisType()->getPointeeType()))
        return transfer(s, call.getArgSVal(0), assign->getCXXThisVal());
    if (const auto *ctor = dyn_cast<CXXConstructorCall>(&call)) {
      if (hostView(ctor->getDecl()->getThisType()->getPointeeType()) && call.getNumArgs() >= 1 && call.getArgExpr(0))
        if (const auto *from = owner(s, call.getArgSVal(0), call.getArgExpr(0)->getType()))
          if (const auto *self = ctor->getCXXThisVal().getAsRegion()) s = s->set<ContainerOwners>(self->StripCasts(), from);
      return s;
    }
    const auto *member = dyn_cast<CXXMemberCall>(&call);
    const auto *method = member ? dyn_cast_or_null<CXXMethodDecl>(member->getDecl()) : nullptr;
    if (!method) return s;
    const bool returnsView = hostView(call.getResultType());  // a conversion such as host_vector::operator host_span
    if (!returnsView && (!method->getIdentifier() || method->getName() != "data" || !call.getResultType()->isPointerType())) return s;
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
      if (auto constructed = call.getReturnValueUnderConstruction()) view = constructed->getAsRegion();
      if (!view) if (auto lazy = call.getReturnValue().getAs<nonloc::LazyCompoundVal>()) view = lazy->getRegion();
      if (!from) return s;
      if (view) return s->set<ContainerOwners>(view->StripCasts(), from);
      return call.getOriginExpr() ? s->set<ViewResults>(call.getOriginExpr(), from) : s;
    }
    const auto *buffer = base(call.getReturnValue());
    if (!from || !buffer) return s;
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
    for (auto op : s->get<MutableReach>()) s = s->remove<MutableReach>(op);
    for (auto op : s->get<ReadOnlyReach>()) s = s->remove<ReadOnlyReach>(op);
    return s;
  }
  void hostAccess(CheckerContext &c, ProgramStateRef s, const MemRegion *r, uint64_t bits, bool isLoad,
                  const char *readMessage, const char *writeMessage) const {
    if (!r) return;
    for (auto op : s->get<PendingOps>()) {
      unsigned overlap = rangeOverlap(s, r, bits, op.region, op.bits);
      if (!overlap) continue;
      if ((isLoad && op.access == HostRead) || (!isLoad && (op.access == HostWrite || op.access == HostRead)))
        report(c, s, 1, isLoad ? readMessage : writeMessage, op.memory, op.memory == Pinned ? "S1" : "S2", &op, overlap != 2);
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
      if (parents.empty()) return true;
      current = parents[0];
      if (current.get<CompoundStmt>() || current.get<SwitchCase>() || current.get<LabelStmt>() ||
          current.get<CXXForRangeStmt>()) return false;
      if (const auto *statement = current.get<IfStmt>()) return statement->getCond() == child;
      if (const auto *statement = current.get<WhileStmt>()) return statement->getCond() == child;
      if (const auto *statement = current.get<DoStmt>()) return statement->getCond() == child;
      if (const auto *statement = current.get<ForStmt>()) return statement->getCond() == child;
      if (const auto *cast = current.get<CastExpr>()) {
        if (cast->getType()->isVoidType()) return false;
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
  void applyCall(const CallEvent &call, CheckerContext &c, ProgramStateRef s, const cudalifetime::Model *given = nullptr) const;
  void applyAllocation(const CallEvent &call, CheckerContext &c, ProgramStateRef s) const;
  void checkLocation(SVal location, bool isLoad, const Stmt *, CheckerContext &c) const;
  void checkBind(SVal location, SVal value, const Stmt *, bool, CheckerContext &c) const;
  void checkPostStmt(const CUDAKernelCallExpr *call, CheckerContext &c) const;
  void checkEndFunction(const ReturnStmt *, CheckerContext &c) const;
  ProgramStateRef checkPointerEscape(ProgramStateRef s, const InvalidatedSymbols &escaped,
       const CallEvent *call, PointerEscapeKind) const {
    // Escapes are counted for coverage only. They are not reported: on an
    // opaque call CSA also invalidates every global, so any buffer reachable
    // from a global would be reported at every such call. Reports use the
    // buffers reachable from the call's own arguments (reach, opaqueCall).
    ++cudalifetime::info().escapes;
    // CSA also emits escape notifications while processing summarized runtime
    // calls. Their pointer effects are handled above, not opaque ownership.
    if (call && !model(name(*call)).effect.empty()) return s;
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
      cudalifetime::info().coverageUnknown.insert("pending region escapes: subsequent accesses through opaque ownership are not modeled");
    }
    return s; // Escape cannot establish completion or retire the operation.
  }
};

void AsyncLifetimeChecker::checkPreCall(const CallEvent &call, CheckerContext &c) const {
  if (model(name(call)).effect.empty()) applyCall(call, c, c.getState());
}

namespace {
// Symbols and regions reachable from a value through memory, as of PreCall.
struct Reachable : SymbolVisitor {
  llvm::SmallPtrSet<SymbolRef, 16> symbols;
  llvm::SmallPtrSet<const MemRegion *, 16> regions;
  bool VisitSymbol(SymbolRef symbol) override { symbols.insert(symbol); return true; }
  bool VisitMemRegion(const MemRegion *region) override { regions.insert(region); return true; }
  bool contains(const MemRegion *r) const {
    if (const auto *symbolic = r->getSymbolicBase(); symbolic && symbols.count(symbolic->getSymbol())) return true;
    for (const auto *region : regions)
      if (region == r || r->isSubRegionOf(region) || region->isSubRegionOf(r)) return true;
    return false;
  }
};
}

// Which pending buffers the call's arguments reach, and whether the callee may
// write them: directly through a parameter (writable unless it points to const),
// through the receiver (unless the method is const), or through pointers stored
// in memory an argument reaches (always writable).
ProgramStateRef AsyncLifetimeChecker::reach(const CallEvent &call, ProgramStateRef s) {
  s = clearReach(s);
  if (s->get<PendingOps>().isEmpty()) return s;
  const auto parameters = call.parameters();
  std::vector<std::pair<const MemRegion *, bool>> direct;  // region, writable
  Reachable indirect;
  for (unsigned i = 0; i < call.getNumArgs(); ++i) {
    const SVal value = call.getArgSVal(i);
    // Variadic and non-pointer parameters give the callee unknown access.
    if (auto *r = base(value)) direct.push_back({r, !(i < parameters.size() && pointsToConst(parameters[i]->getType()))});
    s->scanReachableSymbols(value, indirect);
  }
  if (const auto *method = dyn_cast<CXXInstanceCall>(&call)) {
    const SVal value = method->getCXXThisVal();
    const auto *decl = dyn_cast_or_null<CXXMethodDecl>(call.getDecl());
    if (auto *r = base(value)) direct.push_back({r, !(decl && decl->isConst())});
    s->scanReachableSymbols(value, indirect);
  }
  for (auto op : s->get<PendingOps>()) {
    bool reaches = false, writable = false, viaArgument = false;
    for (const auto &[r, w] : direct)
      if (overlaps(s, r, op.region)) { reaches = viaArgument = true; writable |= w; }
    if (!viaArgument && indirect.contains(op.region)) reaches = writable = true;
    if (reaches) s = writable ? s->add<MutableReach>(op) : s->add<ReadOnlyReach>(op);
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
      if (overlaps(s, base(method->getCXXThisVal()), op.region)) { ++cudalifetime::info().pendingMemberCallUses; break; }
  auto check = [&](const Pending &op, bool mayWrite) {
    if (op.access == DeviceUse) {
      cudalifetime::info().coverageUnknown.insert("pending device region reaches an opaque call: device access effects are unknown");
      return;
    }
    // Through pointers to const the callee can only read, which conflicts
    // only with a pending D2H into the buffer.
    if (mayWrite)
      report(c, s, 1, "Host buffer still used by asynchronous work is passed to an opaque call that may access it", op.memory, "S2", &op, true);
    else if (op.access == HostRead)
      report(c, s, 1, "Opaque call may read a host buffer before asynchronous D2H completion", op.memory, "S2", &op, true);
  };
  for (auto op : writable) check(op, true);
  for (auto op : readOnly) if (!writable.contains(op)) check(op, false);
  return s;
}

void AsyncLifetimeChecker::applyCall(const CallEvent &call, CheckerContext &c, ProgramStateRef s, const cudalifetime::Model *given) const {
  auto n = name(call); auto m = given ? *given : model(n); const auto &e = m.effect;
  Stream stream = streamIdentity(s, arg(s, call, m, "stream"));
  if (m.args.count("stream") && stream == 0) ++cudalifetime::info().unmodeled[n + ":unknown-stream"];
  auto *r = base(arg(s, call, m, "ptr"));
  if (e.empty()) {
    if (cudalifetime::isCudaApi(dyn_cast_or_null<FunctionDecl>(call.getDecl()))) ++cudalifetime::info().unmodeled[n];
    // A lifetime ends here whether or not the callee is inlined. Other calls are
    // checked at PostCall, once it is known whether CSA inlined them.
    if (!endsLifetime(call, n)) {
      auto next = reach(call, s);
      if (next != s) c.addTransition(next);
      return;
    }
    const bool destructor = isa<CXXDestructorCall>(call);
    const MemRegion *self = destructor ? cast<CXXDestructorCall>(call).getCXXThisVal().getAsRegion() : nullptr;
    if (self) self = self->StripCasts();
    for (auto op : s->get<PendingOps>()) {
      bool match = false;
      for (unsigned i = 0; i < call.getNumArgs(); ++i) match |= overlaps(s, base(call.getArgSVal(i)), op.region);
      if (const auto *method = dyn_cast<CXXInstanceCall>(&call)) match |= overlaps(s, base(method->getCXXThisVal()), op.region);
      const MemRegion *const *owned = self ? s->get<ContainerOwners>(op.region->getBaseRegion()) : nullptr;
      if (owned && *owned == self && op.access != DeviceUse) {
        report(c, s, 2, "Host container is destroyed, releasing its buffer, while asynchronous work may still use it", op.memory, "S4", &op, true);
        continue;
      }
      if (match && op.access != DeviceUse)
        report(c, s, 2, destructor ? "Object lifetime ends with pending asynchronous work" : "Host buffer is released with pending asynchronous work",
               op.memory, "S4", &op, true);
    }
    return;
  }
  ++cudalifetime::info().modeledCalls;
  if (capturing(s, stream) && (e == "enqueue_copy" || e == "launch" || e == "memset_async" || e == "free_async")) {
    noteCapture();  // recorded into the graph; nothing runs until the graph is launched
    c.addTransition(s);
    return;
  }
  if (e == "copy") {
    // Synchronous cudaMemcpy runs on the default stream. With host memory on
    // either side it returns only after the copy and all earlier work on the
    // default stream (and on the streams it waits for) is complete, or, for a
    // pageable H2D source, once the source is staged; device-to-device copies
    // do not synchronize. CUDA 13.2.2 archive: api-sync-behavior#memcpy.
    const Stream def = defaultStream();
    auto *dst = base(arg(s, call, m, "dst")), *src = base(arg(s, call, m, "src"));
    uint64_t bits = 0;
    if (auto bytes = arg(s, call, m, "size").getAs<nonloc::ConcreteInt>()) {
      uint64_t count = bytes->getValue()->getLimitedValue();
      if (count <= UINT64_MAX / 8) bits = count * 8;
    }
    int copyKind = -1;
    if (auto v = arg(s, call, m, "kind").getAs<nonloc::ConcreteInt>()) copyKind = v->getValue()->getSExtValue();
    s = enqueue(s, def, c.getSValBuilder().makeZeroVal(c.getASTContext().VoidPtrTy));
    deviceUse(c, s, src, def, "cudaMemcpy"); deviceUse(c, s, dst, def, "cudaMemcpy");
    auto host = [&](const MemRegion *r) { const Memory k = memory(s, r); return k == Pinned || k == Pageable; };
    // With cudaMemcpyDefault and no operand known to be host memory, the copy
    // is treated as device to device; say so when an operand's kind is unknown.
    if (copyKind == 4 && !host(dst) && !host(src) && (memory(s, dst) == Unknown || memory(s, src) == Unknown)) unknownDirection();
    const bool writesHost = copyKind == 0 || copyKind == 2 || (copyKind == 4 && host(dst));
    const bool readsHost = copyKind == 0 || copyKind == 1 || (copyKind == 4 && host(src));
    if (writesHost || readsHost) {
      s = syncStream(s, def);
      // What remains pending is on streams the default stream does not wait for.
      if (writesHost) hostAccess(c, s, dst, bits, false, "", "cudaMemcpy writes a host buffer that asynchronous work on another stream still uses");
      if (readsHost) hostAccess(c, s, src, bits, true, "cudaMemcpy reads a host buffer before an asynchronous D2H on another stream completes", "");
    } else if (copyKind == 3 || copyKind == 4) {
      s = pending(s, dst, def, DeviceUse, call.getOriginExpr(), c);
      s = pending(s, src, def, DeviceUse, call.getOriginExpr(), c);
    } else ++cudalifetime::info().unmodeled[n + ":unknown-copy-kind"];
  } else if (e == "memset" || e == "memset_async") {
    // cudaMemset is asynchronous with respect to the host unless the target is
    // pinned host memory. CUDA 13.2.2 archive: api-sync-behavior#memset.
    const Stream target = e == "memset" ? defaultStream() : stream;
    uint64_t bits = 0;
    if (auto bytes = arg(s, call, m, "size").getAs<nonloc::ConcreteInt>()) {
      uint64_t count = bytes->getValue()->getLimitedValue();
      if (count <= UINT64_MAX / 8) bits = count * 8;
    }
    s = enqueue(s, target, e == "memset" ? c.getSValBuilder().makeZeroVal(c.getASTContext().VoidPtrTy) : arg(s, call, m, "stream"));
    deviceUse(c, s, r, target, e == "memset" ? "cudaMemset" : "cudaMemsetAsync");
    const bool pinned = memory(s, r, bits) == Pinned;
    if (e == "memset" && pinned) {
      s = syncStream(s, target);
      hostAccess(c, s, r, bits, false, "", "cudaMemset writes a host buffer that asynchronous work on another stream still uses");
    } else s = pending(s, r, target, pinned ? HostRead : DeviceUse, call.getOriginExpr(), c, bits);
  } else if (e == "capture_begin") {
    s = s->set<CaptureOrigins>(stream, stream);
    noteCapture();
  } else if (e == "capture_end") {
    for (auto entry : s->get<CaptureOrigins>()) if (entry.second == stream) s = s->remove<CaptureOrigins>(entry.first);
    for (auto entry : s->get<CapturedEvents>()) if (entry.second == stream) s = s->remove<CapturedEvents>(entry.first);
    if (auto out = arg(s, call, m, "ptr").getAs<Loc>())
      s = s->bindLoc(*out, UnknownVal(), c.getStackFrame());
  } else if (e == "graph_launch") {
    s = enqueue(s, stream, arg(s, call, m, "stream"));
    noteCapture();
  } else if (e == "enqueue_copy") {
    s = enqueue(s, stream, arg(s, call, m, "stream"));
    auto *dst = base(arg(s, call, m, "dst")), *src = base(arg(s, call, m, "src"));
    uint64_t bits = 0;
    if (auto bytes = arg(s, call, m, "size").getAs<nonloc::ConcreteInt>()) {
      uint64_t count = bytes->getValue()->getLimitedValue();
      if (!count) { c.addTransition(s); return; }
      if (count <= UINT64_MAX / 8) bits = count * 8;
    }
    int copyKind = -1;
    if (auto v = arg(s, call, m, "kind").getAs<nonloc::ConcreteInt>()) copyKind = v->getValue()->getSExtValue();
    deviceUse(c, s, src, stream, "cudaMemcpyAsync"); deviceUse(c, s, dst, stream, "cudaMemcpyAsync");
    if (copyKind == 1) { s = pending(s, src, stream, HostWrite, call.getOriginExpr(), c, bits); s = pending(s, dst, stream, DeviceUse, call.getOriginExpr(), c); }
    else if (copyKind == 2) { s = pending(s, dst, stream, HostRead, call.getOriginExpr(), c, bits); s = pending(s, src, stream, DeviceUse, call.getOriginExpr(), c); }
    else if (copyKind == 3) { s = pending(s, dst, stream, DeviceUse, call.getOriginExpr(), c); s = pending(s, src, stream, DeviceUse, call.getOriginExpr(), c); }
    else if (copyKind == 4) {
      // cudaMemcpyDefault: unified addressing picks the direction from the
      // pointers. A host destination is written (D2H or H2H), so the host must
      // not touch it; a host source is read (H2D or H2H), so the host must not
      // write it. An operand of unknown memory kind is left out and reported.
      const Memory dstKind = memory(s, dst, bits), srcKind = memory(s, src, bits);
      if (dstKind == Pinned || dstKind == Pageable) s = pending(s, dst, stream, HostRead, call.getOriginExpr(), c, bits);
      else if (dstKind == Device || dstKind == Ordered) s = pending(s, dst, stream, DeviceUse, call.getOriginExpr(), c);
      if (srcKind == Pinned || srcKind == Pageable) s = pending(s, src, stream, HostWrite, call.getOriginExpr(), c, bits);
      else if (srcKind == Device || srcKind == Ordered) s = pending(s, src, stream, DeviceUse, call.getOriginExpr(), c);
      if (dstKind == Unknown || srcKind == Unknown) unknownDirection();
    }
    else if (copyKind != 0) { ++cudalifetime::info().unmodeled[n + ":unknown-copy-kind"]; }
  } else if (e == "launch") {
    s = enqueue(s, stream, arg(s, call, m, "stream"));
    for (unsigned i = 0; i < call.getNumArgs(); ++i) {
      if (m.args.count("stream") && m.args.at("stream") == i) continue;
      auto *buffer = base(call.getArgSVal(i));
      deviceUse(c, s, buffer, stream, "Modeled launch");
      s = pending(s, buffer, stream, DeviceUse, call.getOriginExpr(), c);
    }
  } else if (e == "sync_stream") s = syncStream(s, stream);
  else if (e == "sync_device") {
    for (auto seq : s->get<Sequences>()) s = retire(s, seq.first, seq.second);
  } else if (e == "event_elapsed") {
    // CUDA 13.2.2 event API: an incomplete start or end returns NotReady.
    // A successful timing query therefore proves only their recorded captures.
    for (const char *key : {"start", "end"}) {
      const uint64_t event = eventIdentity(s, arg(s, call, m, key));
      for (auto b : s->get<EventBounds>())
        if (b.first.first == event && b.first.second) s = retire(s, b.first.second, b.second);
    }
    if (auto out = arg(s, call, m, "ms").getAs<Loc>())
      s = s->bindLoc(*out, UnknownVal(), c.getStackFrame());
  } else if (e == "event_record" && capturing(s, stream)) {
    // A record inside a capture is a graph node; a stream that waits on it
    // joins the capture.
    const uint64_t event = eventIdentity(s, arg(s, call, m, "event"));
    s = s->set<EventValues>(event, arg(s, call, m, "event"));
    for (auto b : s->get<EventBounds>()) if (b.first.first == event) s = s->remove<EventBounds>(b.first);
    s = s->set<RecordedEvents>(event, true)->set<CapturedEvents>(event, *s->get<CaptureOrigins>(stream));
  } else if (e == "event_record") {
    const uint64_t event = eventIdentity(s, arg(s, call, m, "event"));
    s = s->set<EventValues>(event, arg(s, call, m, "event"));
    s = enqueue(s, stream, arg(s, call, m, "stream"));
    for (auto b : s->get<EventBounds>()) if (b.first.first == event) s = s->remove<EventBounds>(b.first);
    s = s->set<RecordedEvents>(event, true)->set<EventBounds>({event, stream}, sequence(s, stream));
    for (auto wait : s->get<WaitBounds>()) if (wait.first.first == stream) s = s->set<EventBounds>({event, wait.first.second}, wait.second);
  } else if (e == "stream_wait_event" && (capturing(s, stream) || s->get<CapturedEvents>(eventIdentity(s, arg(s, call, m, "event"))))) {
    if (const auto *origin = s->get<CapturedEvents>(eventIdentity(s, arg(s, call, m, "event"))))
      s = s->set<CaptureOrigins>(stream, *origin);
    noteCapture();
  } else if (e == "stream_wait_event" || e == "sync_event") {
    const uint64_t event = eventIdentity(s, arg(s, call, m, "event"));
    s = s->set<EventValues>(event, arg(s, call, m, "event"));
    const auto *recorded = s->get<RecordedEvents>(event);
    // Only an event created on this path is known to be unrecorded; one read
    // from an object, a container or another process has unknown history.
    if (recorded && !*recorded)
      report(c, s, 4, "Event has no recorded capture on this path; waiting establishes no dependency", Unknown, "S8", nullptr, true);
    else if (!recorded)
      cudalifetime::info().coverageUnknown.insert("event with no creation or record on this path: the dependency it carries is unknown");
    if (e == "stream_wait_event") s = enqueue(s, stream, arg(s, call, m, "stream"));
    for (auto b : s->get<EventBounds>()) if (b.first.first == event) {
      if (e == "sync_event" && b.first.second) s = retire(s, b.first.second, b.second);
      else s = bound(s, stream, b.first.second, b.second);
      // Work after the capture matters only if the waiter is not already ordered after it.
      for (auto alloc : s->get<Allocations>())
        if (!alloc.second.freed && alloc.second.stream == b.first.second && alloc.second.seq > b.second &&
            !ordered(s, stream, alloc.second.stream, alloc.second.seq))
          report(c, s, 4, "Event capture predates asynchronous allocation and does not cover it", Ordered, "S8", nullptr, true);
      for (auto op : s->get<PendingOps>())
        if (op.stream == b.first.second && op.seq > b.second && !ordered(s, stream, op.stream, op.seq))
          report(c, s, 4, "Event capture predates pending producer work and does not cover it", op.memory, "S8", &op, true);
    }
  } else if (e == "free_async" || e == "free_device") {
    if (e == "free_async") s = enqueue(s, stream, arg(s, call, m, "stream"));
    if (const auto *key = allocationRegion(s, r)) {
      const auto *alloc = s->get<Allocations>(key);
      bool reported = false;
      for (auto op : s->get<PendingOps>()) if (overlaps(s, op.region, r)) {
        bool safe = e == "free_async" ? ordered(s, stream, op.stream, op.seq) : completed(s, op.stream) >= op.seq;
        if (!safe) {
          report(c, s, 3, "Stream-ordered allocation is freed while another operation may still use it", Ordered, e == "free_async" ? "S6" : "S7", &op, e == "free_async" && !distinct(s, stream, op.stream));
          reported = true; break;
        }
      }
      if (!reported && e == "free_async" && !ordered(s, stream, alloc->stream, alloc->seq))
        report(c, s, 3, "cudaFreeAsync has no dependency on the asynchronous allocation", Ordered, "S6", nullptr, !distinct(s, stream, alloc->stream), false, alloc, key);
      if (!reported && e == "free_device" && completed(s, alloc->stream) < alloc->seq)
        report(c, s, 3, "Asynchronous allocation is not known complete before cudaFree", Ordered, "S7", nullptr, false, false, alloc, key);
      auto allocation = *alloc; allocation.freed = true;
      allocation.freeStream = stream; allocation.freeSeq = sequence(s, stream);
      s = s->set<Allocations>(key, allocation);
    }
    else if (e == "free_device") {
      for (auto op : s->get<PendingOps>()) if (op.access == DeviceUse && overlaps(s, op.region, r)) {
        report(c, s, 3, "Ordinary cudaFree may implicitly synchronize; this path relies on that behavior", Device, "S10", &op, false, true);
        break;
      }
    }
    // Ordinary cudaFree does not prove unrelated host completion.
  } else if (e == "free_host" || e == "host_unregister") {
    for (auto op : s->get<PendingOps>()) if (overlaps(s, op.region, r) && op.access != DeviceUse)
      report(c, s, 2, "Host buffer is released or unregistered before asynchronous work is known complete", op.memory, "S4", &op, true);
    if (r) s = s->remove<Memories>(r)->remove<RegisteredExtents>(r);
  }
  c.addTransition(s);
}

void AsyncLifetimeChecker::applyAllocation(const CallEvent &call, CheckerContext &c, ProgramStateRef s) const {
  auto m = model(name(call)); const auto &e = m.effect;
  if (e == "host_register") {
    if (auto *r = base(arg(s, call, m, "ptr"))) {
      uint64_t bits = 0;
      if (auto bytes = arg(s, call, m, "size").getAs<nonloc::ConcreteInt>()) {
        uint64_t count = bytes->getValue()->getLimitedValue();
        if (count <= UINT64_MAX / 8) bits = count * 8;
      }
      c.addTransition(s->set<Memories>(r, Pinned)->set<RegisteredExtents>(r, bits));
    }
    return;
  }
  if (e != "malloc_host" && e != "malloc_device" && e != "malloc_async" && e != "stream_create" && e != "event_create") return;
  auto out = arg(s, call, m, "ptr").getAs<Loc>();
  if (!out) return;
  SVal value = UnknownVal();
  {
    auto *origin = call.getOriginExpr();
    if (!origin) return;
    QualType type = c.getASTContext().VoidPtrTy;
    value = c.getSValBuilder().conjureSymbolVal(call, type, c.blockCount());
    s = s->bindLoc(*out, value, c.getStackFrame());
  }
  if (e == "stream_create") {
    unsigned streamKind = name(call) == "cudaStreamCreate" ? 2 : 0;
    if (auto v = arg(s, call, m, "flags").getAs<nonloc::ConcreteInt>()) streamKind = (v->getValue()->getZExtValue() & 1) ? 3 : 2;
    s = s->set<StreamKinds>(identity(value), streamKind)->set<StreamValues>(identity(value), value);
  } else if (e == "event_create") {
    s = s->set<RecordedEvents>(identity(value), false)->set<EventValues>(identity(value), value);
    if (auto flags = arg(s, call, m, "flags").getAs<nonloc::ConcreteInt>())
      s = s->set<EventTimingDisabled>(identity(value), (flags->getValue()->getZExtValue() & 2) != 0);
  } else if (auto *r = base(value)) {
    Memory mem = e == "malloc_host" ? Pinned : e == "malloc_async" ? Ordered : Device;
    s = s->set<Memories>(r, mem);
    if (mem == Ordered) {
      Stream stream = streamIdentity(s, arg(s, call, m, "stream"));
      s = enqueue(s, stream, arg(s, call, m, "stream"));
      s = s->set<Allocations>(r, {stream, sequence(s, stream), false, 0, 0,
                                 call.getOriginExpr(), c.getStackFrame()});
    }
  }
  c.addTransition(s);
}


bool AsyncLifetimeChecker::evalCall(const CallEvent &call, CheckerContext &c) const {
  cudalifetime::info().perThread = c.getPreprocessor().isMacroDefined("CUDA_API_PER_THREAD_DEFAULT_STREAM");
  const auto m = model(name(call));
  if (m.effect.empty() || m.effect == "observe" || !call.getOriginExpr()) return false;
  if (m.effect == "address_only") {  // inspects the pointer value, not the buffer: CSA evaluates it
    if (m.assumption) cudalifetime::info().assumedModels.insert(m.function);
    return false;
  }
  for (const auto &[key, index] : m.args) if (index >= call.getNumArgs()) {
    if (key == "flags" && name(call) == "cudaStreamCreate") continue;
    ++cudalifetime::info().unmodeled[m.function + ":model-argument-out-of-range"]; return false;
  }
  if (m.assumption) if (const auto *fd = dyn_cast_or_null<FunctionDecl>(call.getDecl()))
    if (fd->hasBody()) { ++cudalifetime::info().unmodeled[m.function + ":model-body-deferred-to-csa"]; return false; }
  if (m.assumption) cudalifetime::info().assumedModels.insert(m.function);
  auto s = c.getState();
  bool impossibleSuccess = false;
  if (m.effect == "event_elapsed") for (const char *key : {"start", "end"}) {
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
      if (resultIsUsed(call.getOriginExpr(), c.getASTContext()) || name(call) == "cudaEventQuery" || name(call) == "cudaStreamQuery" || m.effect == "event_elapsed") c.addTransition(failed);
      else {
        ++cudalifetime::info().cudaFailurePathsSkipped;
        c.generateSink(failed, c.getPredecessor());
      }
    }
    if (impossibleSuccess) return true;
    s = s->BindExpr(call.getOriginExpr(), c.getStackFrame(), builder.makeZeroVal(resultType));
  }
  const auto &e = m.effect;
  if (e == "malloc_host" || e == "malloc_device" || e == "malloc_async" ||
      e == "stream_create" || e == "event_create" || e == "host_register")
    applyAllocation(call, c, s);
  else applyCall(call, c, s);
  return true;
}

void AsyncLifetimeChecker::checkLocation(SVal location, bool isLoad, const Stmt *statement, CheckerContext &c) const {
  auto *r = base(location); if (!r) return;
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
  if (!destination || !source || destination->hasMemorySpace<StackSpaceRegion>(s)) return;
  for (auto op : s->get<PendingOps>()) if (overlaps(s, source, op.region)) {
    cudalifetime::info().coverageUnknown.insert("pending region stored outside the local stack: subsequent accesses through opaque ownership are not modeled");
    if (op.access != DeviceUse)
      report(c, s, 1, "Host buffer escapes to nonlocal storage while asynchronous work remains pending", op.memory, "S2", &op, true);
  }
}
void AsyncLifetimeChecker::checkPostStmt(const CUDAKernelCallExpr *call, CheckerContext &c) const {
  ++cudalifetime::info().kernelLaunches;
  auto s = c.getState(); const CallExpr *config = call->getConfig();
  Stream stream = config && config->getNumArgs() > 3 ? streamIdentity(s, kernelValue(config->getArg(3), c)) : (cudalifetime::info().perThread ? 6 : 4);
  if (!stream) ++cudalifetime::info().unmodeled["kernel-launch:unknown-stream"];
  s = enqueue(s, stream, config && config->getNumArgs() > 3 ? kernelValue(config->getArg(3), c) : c.getSValBuilder().makeZeroVal(c.getASTContext().VoidPtrTy));
  for (const auto *argExpr : call->arguments()) {
    const MemRegion *r = base(kernelValue(argExpr, c));
    if (!r && argExpr->getType()->isPointerType()) ++cudalifetime::info().unmodeled["kernel-launch:unknown-pointer-operand"];
    deviceUse(c, s, r, stream, "Kernel launch");
    s = pending(s, r, stream, DeviceUse, call, c);
  }
  c.addTransition(s);
}
void AsyncLifetimeChecker::checkEndFunction(const ReturnStmt *, CheckerContext &c) const {
  auto s = c.getState();
  const auto *frame = c.getStackFrame();
  for (auto op : s->get<PendingOps>()) {
    if (op.access == DeviceUse || !op.region->hasMemorySpace<StackSpaceRegion>(s)) continue;
    if (auto *space = dyn_cast<StackSpaceRegion>(op.region->getMemorySpace(s)))
      if (space->getStackFrame() == frame)
        report(c, s, 2, "Stack buffer lifetime ends while asynchronous work may still access it", op.memory, "S4", &op);
  }
}
}
bool cudalifetime::summarize(const FunctionDecl *decl, Model &out) {
  const auto &summary = AsyncLifetimeChecker::summaryOf(decl);
  if (!summary.copy && !summary.sync) return false;
  out = summary.copy ? *summary.copy : *summary.sync;
  return true;
}
bool cudalifetime::isCudaApi(const FunctionDecl *decl) {
  if (!decl || isa<CXXMethodDecl>(decl) || !decl->getIdentifier() || !decl->getName().starts_with("cu")) return false;
  static const std::set<std::string> headers = {
    "cuda.h", "cuda_runtime.h", "cuda_runtime_api.h", "cuda_device_runtime_api.h", "cudaProfiler.h",
    "cuda_profiler_api.h", "cudaGL.h", "cuda_gl_interop.h", "cudaEGL.h", "cuda_egl_interop.h", "cudaVDPAU.h",
    "cuda_vdpau_interop.h", "cudaD3D9.h", "cudaD3D10.h", "cudaD3D11.h", "cuda_d3d9_interop.h",
    "cuda_d3d10_interop.h", "cuda_d3d11_interop.h"};
  const SourceManager &sources = decl->getASTContext().getSourceManager();
  for (const FunctionDecl *redecl : decl->redecls()) {
    const FunctionDecl *pattern = redecl->getTemplateInstantiationPattern();
    const SourceLocation location = sources.getFileLoc((pattern ? pattern : redecl)->getLocation());
    if (location.isInvalid()) continue;
    if (auto file = sources.getFileEntryRefForID(sources.getFileID(location)))
      if (headers.count(llvm::sys::path::filename(file->getName()).str())) return true;
  }
  return false;
}
void cudalifetime::registerChecker(CheckerRegistry &registry) {
  registry.addChecker<AsyncLifetimeChecker>("cuda.AsyncLifetime", "Checks CUDA host and stream-ordered asynchronous buffer lifetimes", "");
}
