#ifndef CUDA_LIFETIME_ASYNC_LIFETIME_H
#define CUDA_LIFETIME_ASYNC_LIFETIME_H
#include <map>
#include <set>
#include <string>
#include <vector>
namespace clang { class FunctionDecl; namespace ento { class CheckerRegistry; } }
namespace cudalifetime {
struct Model {
  std::string function, effect;
  std::map<std::string, unsigned> args;
  bool assumption = false;
};
// What a report is, carried from the checker to the SARIF consumer as data:
// the consumer looks a diagnostic's bug category up in AnalysisInfo::reports.
struct ReportKind {
  std::string rule, level, memory, semantic;
};
struct AnalysisInfo {
  std::map<std::string, Model> models;
  std::map<std::string, ReportKind> reports;
  std::map<std::string, unsigned> unmodeled;
  std::vector<std::string> budgetExhausted;
  std::set<std::string> assumedModels;
  std::set<std::string> coverageUnknown;
  unsigned kernelLaunches = 0, escapes = 0, modeledCalls = 0, cudaFailurePathsSkipped = 0;
  unsigned pendingMemberCallUses = 0, pendingArgumentUses = 0, pendingPointerEscapes = 0;
  bool perThread = false;
};
AnalysisInfo &info();
bool loadModels(const std::string &directory, std::string &error);
// A CUDA Runtime or Driver API function: a free function named cu... declared
// in one of the toolkit's API headers (by file name, so a stub header with the
// same name also qualifies). Project functions that merely share the prefix are not.
bool isCudaApi(const clang::FunctionDecl *decl);
void registerChecker(clang::ento::CheckerRegistry &registry);
}
#endif
