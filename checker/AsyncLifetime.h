#ifndef CUDA_LIFETIME_ASYNC_LIFETIME_H
#define CUDA_LIFETIME_ASYNC_LIFETIME_H
#include <map>
#include <set>
#include <string>
#include <vector>
namespace clang { namespace ento { class CheckerRegistry; } }
namespace cudalifetime {
struct Model {
  std::string function, effect;
  std::map<std::string, unsigned> args;
  bool assumption = false;
};
struct AnalysisInfo {
  std::map<std::string, Model> models;
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
void registerChecker(clang::ento::CheckerRegistry &registry);
}
#endif
