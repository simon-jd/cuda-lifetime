#include "checker/AsyncLifetime.h"
#include "tool/SarifConsumer.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <sys/resource.h>
using namespace clang;
using namespace clang::tooling;
namespace {
llvm::cl::OptionCategory category("cuda-lifetime-native options");
llvm::cl::opt<std::string> sarif("sarif", llvm::cl::desc("CSA SARIF output file"), llvm::cl::cat(category));
llvm::cl::opt<std::string> coverage("coverage", llvm::cl::desc("Coverage JSON output file"), llvm::cl::cat(category));
llvm::cl::opt<std::string> summarizeTo("summarize", llvm::cl::desc("Write the copy summaries of this TU's function definitions as a model file instead of analyzing it"), llvm::cl::init(""), llvm::cl::cat(category));
llvm::cl::opt<std::string> models("model", llvm::cl::desc("Directory containing model JSON files"), llvm::cl::init(""), llvm::cl::cat(category));
llvm::cl::opt<unsigned> maxNodes("max-nodes", llvm::cl::desc("Maximum exploded nodes per top-level function"), llvm::cl::init(225000), llvm::cl::cat(category));
llvm::cl::opt<bool> nodeProfile("node-profile", llvm::cl::desc("Record where each top-level function's exploded-graph nodes are (by function, direct callee and line) in the coverage file"), llvm::cl::cat(category));
llvm::cl::opt<bool> includeContract("include-contract", llvm::cl::desc("Retain contract diagnostics (filtering is performed by the Python frontend)"), llvm::cl::cat(category));
llvm::cl::opt<std::string> resourceDir("resource-dir", llvm::cl::desc("Clang builtin headers directory; defaults to the matching build SDK"), llvm::cl::init(CUDA_LIFETIME_CLANG_RESOURCE_DIR), llvm::cl::cat(category));
unsigned parsed = 0;
bool skippedNoCuda = false;
std::set<std::string> unsupportedRoots;
// Start at source-file definitions, then follow actual direct callees. Merely
// including a CUDA header must not force analysis of every unused inline body.
class ReachableCuda : public RecursiveASTVisitor<ReachableCuda> {
  std::set<const FunctionDecl *> visited;
  bool auditStaticLocals;
public:
  explicit ReachableCuda(bool audit = true) : auditStaticLocals(audit) {}
  bool found = false;
  void inspect(const FunctionDecl *decl) {
    if (!decl) return;
    std::string qualified = decl->getQualifiedNameAsString();
    if (cudalifetime::isCudaApi(decl) || cudalifetime::info().models.count(qualified)) { found = true; return; }
    if (!visited.insert(decl->getCanonicalDecl()).second) return;
    if (const auto *dtor = dyn_cast<CXXDestructorDecl>(decl)) {
      if (const auto *record = dtor->getParent()->getDefinition()) {
        for (const auto *field : record->fields()) inspectDestructor(field->getType(), decl->getASTContext());
        for (const auto &base : record->bases()) inspectDestructor(base.getType(), decl->getASTContext());
      }
    }
    decl = decl->getDefinition();
    if (!decl) return;
    if (const auto *ctor = dyn_cast<CXXConstructorDecl>(decl))
      for (const auto *init : ctor->inits()) TraverseStmt(init->getInit());
    TraverseStmt(decl->getBody());
  }
  void inspectDestructor(QualType type, ASTContext &ctx) {
    if (const auto *record = ctx.getBaseElementType(type)->getAsCXXRecordDecl()) inspect(record->getDestructor());
  }
  bool VisitCallExpr(CallExpr *call) {
    if (!call->getDirectCallee()) found = true; // An indirect call is not proof of no CUDA use.
    else if (const auto *method = dyn_cast<CXXMethodDecl>(call->getDirectCallee()); method && method->isVirtual()) found = true;
    else inspect(call->getDirectCallee());
    return true;
  }
  bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *) { found = true; return true; }
  bool VisitCXXConstructExpr(CXXConstructExpr *expr) { inspect(expr->getConstructor()); return true; }
  bool VisitVarDecl(VarDecl *decl) {
    inspectDestructor(decl->getType(), decl->getASTContext());
    if (auditStaticLocals && decl->isStaticLocal()) {
      ReachableCuda destruction(false);
      destruction.inspectDestructor(decl->getType(), decl->getASTContext());
      if (destruction.found)
        unsupportedRoots.insert("CSA does not guarantee exploration of static local destruction: " + decl->getQualifiedNameAsString());
    }
    return true;
  }
  bool VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *expr) { inspect(expr->getTemporary()->getDestructor()); return true; }
  bool VisitCXXDeleteExpr(CXXDeleteExpr *expr) {
    if (!expr->getDestroyedType().isNull())
      if (const auto *record = expr->getDestroyedType()->getAsCXXRecordDecl()) inspect(record->getDestructor());
    return true;
  }
};
class SourceRoots : public RecursiveASTVisitor<SourceRoots> {
  SourceManager &sources;
public:
  ReachableCuda scan;
  explicit SourceRoots(SourceManager &sources) : sources(sources) {}
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool inMainFile(SourceLocation loc) const { return loc.isValid() && sources.isWrittenInMainFile(sources.getExpansionLoc(loc)); }
  bool VisitFunctionDecl(FunctionDecl *decl) {
    if (decl->doesThisDeclarationHaveABody() && inMainFile(decl->getLocation())) scan.inspect(decl);
    if (decl->getTemplateSpecializationKind() == TSK_ExplicitInstantiationDefinition &&
        inMainFile(decl->getPointOfInstantiation()) && !inMainFile(decl->getLocation())) {
      ReachableCuda instance;
      instance.inspect(decl);
      if (instance.found) {
        scan.found = true;
        unsupportedRoots.insert("CSA does not guarantee exploration of a header-defined explicit template instantiation: " + decl->getQualifiedNameAsString());
      }
    }
    return true;
  }
  bool VisitVarDecl(VarDecl *decl) {
    // A dependent template member/variable is not a dynamic object in this TU.
    // Its actual specializations are visited independently after instantiation.
    if (decl->getDeclContext()->isDependentContext() || decl->getType()->isDependentType() ||
        decl->getDescribedVarTemplate() ||
        (decl->hasInit() && (decl->getInit()->isTypeDependent() || decl->getInit()->isValueDependent()))) return true;
    if (decl->hasGlobalStorage() && !decl->isStaticLocal()) {
      ReachableCuda global;
      if (decl->hasInit()) global.TraverseStmt(decl->getInit());
      // An extern declaration does not register a destructor in this TU.
      if (decl->isThisDeclarationADefinition() != VarDecl::DeclarationOnly)
        global.inspectDestructor(decl->getType(), decl->getASTContext());
      if (global.found) {
        scan.found = true;
        unsupportedRoots.insert("CSA does not guarantee exploration of global initialization or destruction: " + decl->getQualifiedNameAsString());
      }
    }
    return true;
  }
};
class SelectiveAnalysis : public ASTConsumer {
  std::unique_ptr<ASTConsumer> analysis;
public:
  explicit SelectiveAnalysis(std::unique_ptr<ASTConsumer> consumer) : analysis(std::move(consumer)) {}
  void Initialize(ASTContext &ctx) override { analysis->Initialize(ctx); }
  bool HandleTopLevelDecl(DeclGroupRef group) override { return analysis->HandleTopLevelDecl(group); }
  void HandleTopLevelDeclInObjCContainer(DeclGroupRef group) override { analysis->HandleTopLevelDeclInObjCContainer(group); }
  void HandleTranslationUnit(ASTContext &ctx) override {
    SourceRoots roots(ctx.getSourceManager());
    roots.TraverseDecl(ctx.getTranslationUnitDecl());
    skippedNoCuda = !roots.scan.found;
    if (!skippedNoCuda) analysis->HandleTranslationUnit(ctx);
  }
};
// Summaries for other translation units: every function defined in the main
// file whose body queues a copy on its parameters (cudalifetime::summarize),
// written in the model format. Names defined more than once (overloads) are
// left out, since models are looked up by qualified name.
class Summaries : public ASTConsumer, public RecursiveASTVisitor<Summaries> {
  SourceManager *sources = nullptr;
  std::map<std::string, unsigned> definitions;
  std::map<std::string, std::pair<cudalifetime::Model, unsigned>> found;
public:
  bool VisitFunctionDecl(FunctionDecl *decl) {
    if (!decl->doesThisDeclarationHaveABody() || decl->isDependentContext() || !decl->getIdentifier() || !decl->isExternallyVisible()) return true;
    if (!sources->isWrittenInMainFile(sources->getExpansionLoc(decl->getLocation()))) return true;
    std::string qualified = decl->getQualifiedNameAsString();
    if (cudalifetime::info().models.count(qualified)) return true;
    ++definitions[qualified];
    cudalifetime::Model model;
    if (cudalifetime::summarize(decl, model))
      found[qualified] = {model, sources->getExpansionLineNumber(decl->getLocation())};
    return true;
  }
  void HandleTranslationUnit(ASTContext &ctx) override {
    sources = &ctx.getSourceManager();
    TraverseDecl(ctx.getTranslationUnitDecl());
    llvm::SmallString<256> path(sources->getFileEntryRefForID(sources->getMainFileID())->getName());
    llvm::sys::fs::make_absolute(path);
    llvm::json::Array models;
    for (const auto &[name, entry] : found) {
      if (definitions[name] != 1) continue;
      llvm::json::Object args, constants;
      for (const auto &[key, index] : entry.first.args) args[key] = index;
      for (const auto &[key, value] : entry.first.constants) constants[key] = value;
      llvm::json::Array elements;
      for (const auto &key : entry.first.elements) elements.push_back(key);
      llvm::json::Array receiver;
      for (const auto &key : entry.first.receiver) receiver.push_back(key);
      models.push_back(llvm::json::Object{{"function", name}, {"aliases", llvm::json::Array{}}, {"effect", entry.first.effect},
          {"args", std::move(args)}, {"constants", std::move(constants)}, {"elements", std::move(elements)}, {"receiver", std::move(receiver)}, {"assumption", true}, {"generated", true},
          {"rationale", entry.first.effect == "enqueue_copy" ? "Generated from the function body: every copy it queues, through the calls it makes, forwards these parameters; it calls no synchronization." : "Generated from the function body: every path that returns normally passes this synchronization."},
          {"doc", llvm::json::Object{{"url", "file://" + path.str().str()}, {"anchor", "L" + std::to_string(entry.second)}, {"cudaVersion", "13.2.2"}}}});
    }
    std::error_code ec;
    llvm::raw_fd_ostream out(summarizeTo, ec);
    if (ec) { llvm::errs() << "Cannot write summaries: " << ec.message() << "\n"; return; }
    out << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{{"schemaVersion", 1}, {"generatedFrom", path.str().str()}, {"models", std::move(models)}})) << "\n";
  }
};
class SummaryAction : public ASTFrontendAction {
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &, llvm::StringRef) override { return std::make_unique<Summaries>(); }
};
class TrackingDiagnostics : public TextDiagnosticPrinter {
public:
  bool sawError = false;
  explicit TrackingDiagnostics(DiagnosticOptions &opts) : TextDiagnosticPrinter(llvm::errs(), opts) {}
  void HandleDiagnostic(DiagnosticsEngine::Level level, const Diagnostic &diagnostic) override {
    sawError |= level >= DiagnosticsEngine::Error;
    TextDiagnosticPrinter::HandleDiagnostic(level, diagnostic);
  }
};
class Action : public ASTFrontendAction {
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci, llvm::StringRef) override {
    auto &opts = ci.getAnalyzerOpts();
    opts.CheckersAndPackages.emplace_back("cuda.AsyncLifetime", true);
    opts.AnalysisDiagOpt = PD_NONE;
    opts.MaxNodesPerTopLevelFunction = maxNodes;
    // Profiling counts every node created: no reclamation of uninteresting ones.
    if (nodeProfile) opts.GraphTrimInterval = 0;
    ci.getFrontendOpts().OutputFile = sarif;
    auto consumer = ento::CreateAnalysisConsumer(ci);
    consumer->AddCheckerRegistrationFn(cudalifetime::registerChecker);
    consumer->AddDiagnosticConsumer(cudalifetime::makeSarifConsumer(ci, sarif));
    return std::make_unique<SelectiveAnalysis>(std::move(consumer));
  }
  void ExecuteAction() override {
    cudalifetime::info().perThread = getCompilerInstance().getLangOpts().GPUDefaultStream == LangOptions::GPUDefaultStreamKind::PerThread || getCompilerInstance().getPreprocessor().isMacroDefined("CUDA_API_PER_THREAD_DEFAULT_STREAM");
    ASTFrontendAction::ExecuteAction();
  }
  void EndSourceFileAction() override {
    if (!getCompilerInstance().getDiagnostics().hasErrorOccurred()) ++parsed;
  }
};
}
int main(int argc, const char **argv) {
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &out) { out << "cuda-lifetime-native " CUDA_LIFETIME_PRODUCT_VERSION " (LLVM " CUDA_LIFETIME_LLVM_VERSION ")\n"; });
  auto options = CommonOptionsParser::create(argc, argv, category, llvm::cl::OneOrMore);
  if (!options) { llvm::errs() << llvm::toString(options.takeError()) << "\n"; return 2; }
  if (options->getSourcePathList().size() != 1) { llvm::errs() << "Run one TU per process to isolate SARIF and budgets; use cuda-lifetime analyze for batches.\n"; return 2; }
  if (!maxNodes) { llvm::errs() << "--max-nodes must be positive\n"; return 2; }
  if (summarizeTo.empty() && (sarif.empty() || coverage.empty())) { llvm::errs() << "--sarif and --coverage are required unless --summarize is given\n"; return 2; }
  cudalifetime::info().profileNodes = nodeProfile;
  std::string error;
  if (!cudalifetime::loadModels(models, error)) { llvm::errs() << error << "\n"; return 2; }
  ClangTool tool(options->getCompilations(), options->getSourcePathList());
  if (!llvm::sys::fs::is_directory(resourceDir)) { llvm::errs() << "Clang resource directory is missing; supply --resource-dir for LLVM " CUDA_LIFETIME_LLVM_VERSION "\n"; return 2; }
  tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(("-resource-dir=" + resourceDir.getValue()).c_str(), ArgumentInsertPosition::BEGIN));
  DiagnosticOptions diagnosticOptions;
  TrackingDiagnostics diagnostics(diagnosticOptions);
  tool.setDiagnosticConsumer(&diagnostics);
  if (!summarizeTo.empty()) {
    int status = tool.run(newFrontendActionFactory<SummaryAction>().get());
    return diagnostics.sawError ? 1 : status;
  }
  auto start = std::chrono::steady_clock::now();
  int status = tool.run(newFrontendActionFactory<Action>().get());
  if (diagnostics.sawError) { status = 1; parsed = 0; }
  if (skippedNoCuda && !status) {
    // A skipped invocation still returns a valid, attributable empty SARIF run.
    llvm::json::Object empty{{"version", "2.1.0"}, {"runs", llvm::json::Array{
        llvm::json::Object{{"tool", llvm::json::Object{{"driver", llvm::json::Object{{"name", "cuda-lifetime"}, {"version", CUDA_LIFETIME_PRODUCT_VERSION}}}}},
                           {"results", llvm::json::Array{}}, {"properties", llvm::json::Object{{"skipReason", "no reachable CUDA calls in this TU or declared models"}}}}}}};
    std::error_code error;
    llvm::raw_fd_ostream out(sarif, error);
    if (error) { llvm::errs() << "Cannot write SARIF: " << error.message() << "\n"; status = 2; }
    else out << llvm::formatv("{0:2}", llvm::json::Value(std::move(empty))) << "\n";
  }
  double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  llvm::json::Array failed, analyzed, exhausted;
  const std::string &source = options->getSourcePathList().front();
  if (status || !parsed) failed.push_back(llvm::json::Object{{"file", source}, {"reason", "Clang parsing or analysis failed"}});
  else analyzed.push_back(source);
  llvm::json::Array entryPoints;
  uint64_t totalNodes = 0;
  for (const auto &entry : cudalifetime::info().entryPoints) {
    totalNodes += entry.nodes;
    llvm::json::Object record{{"function", entry.function}, {"nodes", entry.nodes},
                              {"milliseconds", entry.milliseconds}, {"exhausted", entry.exhausted}};
    auto counts = [](const std::vector<std::pair<std::string, unsigned>> &pairs) {
      llvm::json::Array out;
      for (const auto &[label, count] : pairs) out.push_back(llvm::json::Array{label, count});
      return out;
    };
    if (nodeProfile)
      record["profile"] = llvm::json::Object{{"byFrame", counts(entry.byFrame)}, {"byCallee", counts(entry.byCallee)},
                                             {"byLine", counts(entry.byLine)}, {"bySite", counts(entry.bySite)}};
    entryPoints.push_back(std::move(record));
  }
  for (auto &function : cudalifetime::info().budgetExhausted)
    exhausted.push_back(llvm::json::Object{{"file",source},{"function",function},{"reason","CSA work remains: node budget, loop limit, or aborted block"}});
  llvm::json::Object unmodeled;
  llvm::json::Array coverageUnknown;
  for (const auto &reason : unsupportedRoots)
    coverageUnknown.push_back(llvm::json::Object{{"file", source}, {"reason", reason}});
  for (const auto &reason : cudalifetime::info().coverageUnknown)
    coverageUnknown.push_back(llvm::json::Object{{"file", source}, {"reason", reason}});
  for (const auto &[name, count] : cudalifetime::info().unmodeled) unmodeled[name] = count;
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  llvm::json::Array assumptions{"single host thread", "single device", "CUDA calls with discarded return values are assumed successful; explicitly checked failure paths are explored"};
  for (const auto &model : cudalifetime::info().assumedModels) assumptions.push_back("user model: " + model);
  llvm::json::Object result{
      {"schemaVersion",1}, {"llvmVersion",CUDA_LIFETIME_LLVM_VERSION},
      {"astScan", llvm::json::Object{{"skippedNoCudaCalls", skippedNoCuda}, {"scope", "main-file definitions/instantiations/globals, direct callees and implicit constructor/destructor calls; indirect calls retain CSA analysis"}}},
      {"tus",llvm::json::Object{{"parsed",parsed},{"analyzed",std::move(analyzed)},{"failed",std::move(failed)},{"translationUnknown",llvm::json::Array{}}}},
      {"budgetExhausted",std::move(exhausted)}, {"budgetTracking","CSA EndAnalysis hasWorkRemaining"},
      {"coverageUnknown",std::move(coverageUnknown)},
      {"unmodeledCudaCalls",std::move(unmodeled)}, {"maxNodes",maxNodes.getValue()},
      {"kernelLaunches",cudalifetime::info().kernelLaunches}, {"pointerEscapes",cudalifetime::info().escapes},
      {"pendingMemberCallUses",cudalifetime::info().pendingMemberCallUses},
      {"pendingArgumentUses",cudalifetime::info().pendingArgumentUses},
      {"pendingPointerEscapes",cudalifetime::info().pendingPointerEscapes},
      {"modeledCallVisits",cudalifetime::info().modeledCalls},
      {"functionsAnalyzed",cudalifetime::info().functionsAnalyzed},
      {"nodes",static_cast<int64_t>(totalNodes)}, {"nodesCounted", nodeProfile ? "all created" : "kept after reclamation"},
      {"entryPoints",std::move(entryPoints)},
      {"cudaFailurePathsSkipped",cudalifetime::info().cudaFailurePathsSkipped},
      {"pathsEndedBeforeThrow",cudalifetime::info().pathsEndedBeforeThrow},
      {"libraryCallsNotInlined",cudalifetime::info().libraryCallsNotInlined},
      {"defaultStreamMode",cudalifetime::info().perThread ? "per-thread" : "legacy"},
      {"seconds",seconds}, {"peakRssKiB",static_cast<int64_t>(usage.ru_maxrss)}, {"assumptions",std::move(assumptions)}};
  std::error_code ec;
  llvm::raw_fd_ostream out(coverage, ec);
  if (ec) { llvm::errs() << "Cannot write coverage: " << ec.message() << "\n"; return 2; }
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(result))) << "\n";
  return status;
}
