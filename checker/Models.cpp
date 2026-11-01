#include "checker/AsyncLifetime.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <system_error>
#include <set>
#include <limits>
namespace cudalifetime {
AnalysisInfo &info() { static AnalysisInfo value; return value; }
static bool loadEntry(const llvm::json::Value &entry, std::string &error) {
  const auto *obj = entry.getAsObject();
  if (!obj) { error = "model entry must be an object"; return false; }
  auto function = obj->getString("function"), effect = obj->getString("effect");
  if (!function || function->empty() || !effect || effect->empty()) { error = "model requires nonempty function and effect"; return false; }
  for (const char *flag : {"assumption", "userAssumption"})
    if (obj->get(flag) && !obj->getBoolean(flag)) { error = std::string("model ") + flag + " must be boolean"; return false; }
  if (!obj->getArray("aliases")) { error = "model requires aliases array"; return false; }
  const auto *doc = obj->getObject("doc");
  if (!doc || !doc->getString("url") || doc->getString("url")->empty() || !doc->getString("anchor") || doc->getString("anchor")->empty() ||
      !doc->getString("cudaVersion") || *doc->getString("cudaVersion") != "13.2.2") {
    error = "model requires doc url, anchor, and cudaVersion 13.2.2"; return false;
  }
  Model m; m.function = function->str(); m.effect = effect->str();
  m.assumption = true; // Every external summary is a disclosed user assumption.
  static const std::map<std::string, std::set<std::string>> required = {
    {"enqueue_copy", {"stream"}}, {"launch", {"stream"}},
    {"malloc_host", {"ptr"}}, {"host_register", {"ptr"}}, {"host_unregister", {"ptr"}},
    {"malloc_device", {"ptr"}}, {"malloc_async", {"ptr", "stream"}}, {"free_async", {"ptr", "stream"}},
    {"free_host", {"ptr"}}, {"free_device", {"ptr"}}, {"sync_device", {}}, {"sync_stream", {"stream"}},
    {"sync_event", {"event"}}, {"event_elapsed", {"start", "end"}}, {"event_record", {"event", "stream"}},
    {"stream_wait_event", {"stream", "event"}}, {"stream_create", {"ptr"}}, {"event_create", {"ptr"}}, {"destroy", {}},
    {"copy", {"dst", "src"}}, {"memset", {"ptr"}}, {"memset_async", {"ptr", "stream"}}, {"address_only", {}},
    {"capture_begin", {"stream"}}, {"capture_end", {"stream"}}, {"graph_launch", {"stream"}}
  };
  if (!required.count(m.effect)) { error = "unsupported model effect: " + m.effect; return false; }
  if (!obj->getObject("args")) { error = "model requires args object"; return false; }
  if (const auto *args = obj->getObject("args")) {
    for (const auto &arg : *args) {
      auto index = arg.second.getAsInteger();
      if (!index || *index < 0 || static_cast<uint64_t>(*index) > std::numeric_limits<unsigned>::max()) { error = "model argument indices must be nonnegative integers"; return false; }
      m.args[arg.first.str()] = static_cast<unsigned>(*index);
    }
  }
  if (const auto *constants = obj->getObject("constants")) {
    for (const auto &constant : *constants) {
      auto value = constant.second.getAsInteger();
      const std::string key = constant.first.str();
      if (!value || (key != "kind" && key != "size" && key != "stream")) { error = "model constants must be integers for kind, size or stream"; return false; }
      if (m.args.count(key)) { error = "model " + key + " is both an argument and a constant: " + m.function; return false; }
      m.constants[constant.first.str()] = *value;
    }
  }
  if (const auto *elements = obj->getArray("elements")) {
    for (const auto &element : *elements) {
      auto key = element.getAsString();
      if (!key || !m.args.count(key->str())) { error = "model elements must name arguments: " + m.function; return false; }
      m.elements.insert(key->str());
    }
  }
  for (const auto &key : required.at(m.effect)) if (!m.args.count(key) && !m.constants.count(key)) { error = "missing argument " + key + " for model " + m.function; return false; }
  // A copy summary needs the host side of the copy among its arguments; the
  // other side may belong to the receiver (for example a device buffer member).
  if (m.effect == "enqueue_copy" && !m.args.count("dst") && !m.args.count("src")) { error = "missing argument dst or src for model " + m.function; return false; }
  if (info().models.count(m.function)) { error = "duplicate model function: " + m.function; return false; }
  info().models[m.function] = m;
  if (const auto *aliases = obj->getArray("aliases")) {
    for (const auto &alias : *aliases) {
      auto name = alias.getAsString();
      if (!name || name->empty()) { error = "model alias must be a nonempty string"; return false; }
      if (info().models.count(name->str())) { error = "duplicate model alias: " + name->str(); return false; }
      info().models[name->str()] = m;
    }
  }
  return true;
}
bool loadModels(const std::string &directory, std::string &error) {
  if (directory.empty()) return true;
  std::error_code ec;
  llvm::sys::fs::directory_iterator it(directory, ec), end;
  if (ec) { error = "cannot open model directory: " + ec.message(); return false; }
  for (; it != end; it.increment(ec)) {
    if (ec) { error = ec.message(); return false; }
    if (llvm::sys::path::extension(it->path()) != ".json") continue;
    auto file = llvm::MemoryBuffer::getFile(it->path());
    if (!file) { error = "cannot read " + it->path(); return false; }
    auto parsed = llvm::json::parse((*file)->getBuffer());
    if (!parsed) { error = "invalid model JSON: " + llvm::toString(parsed.takeError()); return false; }
    const llvm::json::Array *entries = parsed->getAsArray();
    if (!entries) if (auto *o = parsed->getAsObject()) {
      if (o->getInteger("schemaVersion") != 1) { error = "model schemaVersion must be integer 1"; return false; }
      entries = o->getArray("models");
    }
    if (!entries) { error = "model file must be an array or contain models array: " + it->path(); return false; }
    for (const auto &entry : *entries) if (!loadEntry(entry, error)) return false;
  }
  return true;
}
}
