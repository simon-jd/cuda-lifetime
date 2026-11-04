#ifndef CUDA_LIFETIME_ASYNC_LIFETIME_H
#define CUDA_LIFETIME_ASYNC_LIFETIME_H
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace clang {
class FunctionDecl;
namespace ento {
class CheckerRegistry;
}
} // namespace clang
namespace cudalifetime {
struct Model {
  std::string function, effect;
  std::map<std::string, unsigned> args;
  // Fixed argument values, such as the copy kind of a wrapper that always
  // passes cudaMemcpyDefault.
  std::map<std::string, int64_t> constants;
  // Arguments that are arrays whose first element is the operand (batched copies).
  std::set<std::string> elements;
  // Keys whose operand is the receiver of a member call (a stream wrapper's synchronize()).
  std::set<std::string> receiver;
  bool assumption = false;
};
// What a report is, carried from the checker to the SARIF consumer as data:
// the consumer looks a diagnostic's bug category up in AnalysisInfo::reports.
struct ReportKind {
  std::string rule, level, memory, semantic;
};
// One top-level function's exploration: its exploded-graph nodes at the end
// of the analysis (CSA's node budget counts the work-list steps that create
// them) and its path-analysis time. CSA reclaims uninteresting nodes as it
// goes, so nodes are those kept, except with --node-profile, which turns
// reclamation off and counts every node created. The profile records where
// the nodes are: by innermost function, by the top-level function's direct
// callee they descend from, by source line, and by the project
// (non-system-header) line they are under.
struct EntryPoint {
  std::string function;
  unsigned nodes = 0, milliseconds = 0;
  bool exhausted = false;
  std::vector<std::pair<std::string, unsigned>> byFrame, byCallee, byLine, bySite;
};
struct AnalysisInfo {
  std::map<std::string, Model> models;
  std::map<std::string, ReportKind> reports;
  std::map<std::string, unsigned> unmodeled;
  std::vector<std::string> budgetExhausted;
  std::set<std::string> assumedModels;
  std::set<std::string> coverageUnknown;
  unsigned kernelLaunches = 0, escapes = 0, modeledCalls = 0, cudaFailurePathsSkipped = 0;
  // Paths ended on entering a block from which every path throws or calls a function that never returns.
  unsigned pathsEndedBeforeThrow = 0;
  unsigned pendingMemberCallUses = 0, pendingArgumentUses = 0, pendingPointerEscapes = 0;
  // Top-level functions CSA analyzed: the denominator for budgetExhausted.
  unsigned functionsAnalyzed = 0;
  std::vector<EntryPoint> entryPoints;
  bool profileNodes = false;
  bool perThread = false;
};
AnalysisInfo &info();
bool loadModels(const std::string &directory, std::string &error);
// A CUDA Runtime or Driver API function: a free function named cu... declared
// in one of the toolkit's API headers (by file name, so a stub header with the
// same name also qualifies). Project functions that merely share the prefix are not.
bool isCudaApi(const clang::FunctionDecl *decl);
// The copy a function's body queues on its own parameters, or failing that
// the synchronization it performs on every path, derived from its AST
// through the calls it makes (CUDA calls, models, other bodies).
bool summarize(const clang::FunctionDecl *decl, Model &out);
void registerChecker(clang::ento::CheckerRegistry &registry);
} // namespace cudalifetime
#endif
