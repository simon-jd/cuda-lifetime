#include "tool/SarifConsumer.h"
#include "checker/AsyncLifetime.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <set>

using namespace clang;
using namespace clang::ento;
namespace {
bool restoreArtifactOrder(llvm::json::Object &document) {
  auto *runs = document.getArray("runs");
  if (!runs) return false;
  for (auto &run : *runs) {
    auto *object = run.getAsObject();
    auto *artifacts = object ? object->getArray("artifacts") : nullptr;
    if (!artifacts) return false;
    std::set<int64_t> indices;
    for (const auto &artifact : *artifacts) {
      const auto *entry = artifact.getAsObject();
      const auto *location = entry ? entry->getObject("location") : nullptr;
      const auto index = location ? location->getInteger("index") : std::nullopt;
      if (!index || *index < 0 || static_cast<uint64_t>(*index) >= artifacts->size() ||
          !indices.insert(*index).second) return false;
    }
    // LLVM 23 assigns indices in insertion order but serializes artifacts in
    // URI order. Restore the validated permutation; leave every result and
    // path reference, message and source coordinate exactly as CSA emitted it.
    std::sort(artifacts->begin(), artifacts->end(), [](const auto &a, const auto &b) {
      return *a.getAsObject()->getObject("location")->getInteger("index") <
             *b.getAsObject()->getObject("location")->getInteger("index");
    });
  }
  return true;
}
// LLVM 23's stock SARIF consumer looks up descriptions in the built-in
// Checkers.inc table and llvm_unreachable()s for externally registered checkers.
// Use its public SARIF object model with the real CSA PathDiagnostic instead.
class SarifConsumer final : public PathDiagnosticConsumer {
  CompilerInstance &compiler;
  std::string output;

  std::optional<CharSourceRange> characterRange(const PathDiagnosticLocation &location) const {
    auto &sources = compiler.getSourceManager();
    SourceLocation begin, end;
    if (location.isValid()) {
      SourceRange original = location.asRange();
      if (original.isValid()) {
        auto expanded = sources.getExpansionRange(original);
        begin = expanded.getBegin();
        end = expanded.getEnd();
        if (begin != end && expanded.isTokenRange())
          end = Lexer::getLocForEndOfToken(end, 0, sources, compiler.getLangOpts());
      }
      if (begin.isInvalid()) begin = sources.getExpansionLoc(location.asLocation());
    }
    if (begin.isInvalid() || !sources.getFileEntryRefForID(sources.getFileID(begin))) {
      // Preserve the result message, fail coverage, and never fabricate a
      // source coordinate when CSA itself supplies no file-backed anchor.
      cudalifetime::info().coverageUnknown.insert("CSA SARIF path location has no file-backed expansion anchor");
      auto &engine = compiler.getDiagnostics();
      engine.Report(engine.getCustomDiagID(DiagnosticsEngine::Error,
          "cannot serialize a CSA path location without a file-backed expansion anchor"));
      return std::nullopt;
    }
    // Some macro/token ranges have no representable final token. A caret at
    // this diagnostic's real expansion anchor remains an accurate location.
    if (end.isInvalid() || sources.getFileID(begin) != sources.getFileID(end) ||
        sources.getFileOffset(end) < sources.getFileOffset(begin)) end = begin;
    return CharSourceRange::getCharRange(begin, end);
  }

public:
  SarifConsumer(CompilerInstance &compiler, std::string output)
      : compiler(compiler), output(std::move(output)) {}
  llvm::StringRef getName() const override { return "CudaLifetimeSarif"; }
  PathGenerationScheme getGenerationScheme() const override { return Minimal; }
  bool supportsLogicalOpControlFlow() const override { return true; }
  bool supportsCrossFileDiagnostics() const override { return true; }

  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &diagnostics,
                            FilesMade *) override {
    std::error_code error;
    llvm::raw_fd_ostream stream(output, error);
    auto outputError = [&] {
      auto &engine = compiler.getDiagnostics();
      engine.Report(engine.getCustomDiagID(DiagnosticsEngine::Error,
                     "cannot write CUDA lifetime SARIF output '%0'")) << output;
    };
    if (error) { outputError(); return; }

    SarifDocumentWriter writer(compiler.getSourceManager());
    // createRun retains its StringRef in the JSON value until serialization.
    const std::string toolVersion = std::string(CUDA_LIFETIME_PRODUCT_VERSION) + " (" + getClangFullVersion() + ")";
    writer.createRun("cuda-lifetime-native", "Clang Static Analyzer: CUDA asynchronous lifetime",
                     toolVersion);
    llvm::StringMap<uint32_t> rules;
    for (const auto *diagnostic : diagnostics) {
      llvm::StringRef checker = diagnostic->getCheckerName();
      auto [entry, inserted] = rules.try_emplace(checker, 0);
      if (inserted) {
        auto rule = SarifRule::create().setRuleId(checker).setName(checker)
            .setDescription("CUDA asynchronous host and stream-ordered allocation lifetime");
        entry->second = writer.createRule(rule);
      }
      llvm::SmallVector<ThreadFlow, 8> flows;
      for (const auto &piece : diagnostic->path.flatten(false)) {
        auto range = characterRange(piece->getLocation());
        if (!range) continue;
        auto importance = piece->getKind() == PathDiagnosticPiece::Event
            ? ThreadFlowImportance::Essential : ThreadFlowImportance::Unimportant;
        flows.push_back(ThreadFlow::create().setRange(*range)
            .setImportance(importance).setMessage(piece->getString()));
      }
      auto result = SarifResult::create(entry->second).setRuleId(checker)
          .setDiagnosticMessage(diagnostic->getVerboseDescription())
          .setDiagnosticLevel(SarifResultLevel::Warning)
          .addPartialFingerprint("clang/issueHash/v1", diagnostic->getIssueHash(
              compiler.getSourceManager(), compiler.getLangOpts()))
          .setThreadFlows(flows);
      if (auto range = characterRange(diagnostic->getLocation())) result = result.addLocations({*range});
      writer.appendResult(result);
    }
    auto document = writer.createDocument();
    if (!restoreArtifactOrder(document)) {
      cudalifetime::info().coverageUnknown.insert("SARIF writer artifact indices are not a complete unique permutation");
      auto &engine = compiler.getDiagnostics();
      engine.Report(engine.getCustomDiagID(DiagnosticsEngine::Error,
          "cannot normalize SARIF artifact indices: expected a unique 0..N-1 permutation"));
    }
    stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(document)));
    stream.flush();
    if (stream.has_error()) { stream.clear_error(); outputError(); }
  }
};
}
std::unique_ptr<PathDiagnosticConsumer>
cudalifetime::makeSarifConsumer(CompilerInstance &compiler, const std::string &output) {
  return std::make_unique<SarifConsumer>(compiler, output);
}
