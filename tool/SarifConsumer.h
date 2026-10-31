#ifndef CUDA_LIFETIME_SARIF_CONSUMER_H
#define CUDA_LIFETIME_SARIF_CONSUMER_H
#include <memory>
#include <string>
namespace clang {
class CompilerInstance;
namespace ento { class PathDiagnosticConsumer; }
}
namespace cudalifetime {
std::unique_ptr<clang::ento::PathDiagnosticConsumer>
makeSarifConsumer(clang::CompilerInstance &compiler, const std::string &output);
}
#endif
