"""Write-once hashes for labels and sources; verification never rewrites labels."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .metrics import EvaluationError, _cases, normalized_file


def _hash(path):
    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationError(f"cannot hash artifact: {path}") from exc


def create_freeze(labels_path, manifest_path, root=".", rule_files=()):
    """Capture actual current bytes, failing if a freeze already exists."""
    labels_path, manifest_path = Path(labels_path), Path(manifest_path)
    labels = json.loads(labels_path.read_text())
    cases = _cases(labels, root)
    if labels.get("split") == "holdout" and not rule_files:
        raise EvaluationError("a holdout freeze requires explicitly recorded rule files")
    sources = sorted({case["file"] for case in cases})
    root_path = Path(root).resolve()

    def entry(path):
        path = Path(path).resolve()
        try:
            relative = str(path.relative_to(root_path))
        except ValueError as exc:
            raise EvaluationError("frozen artifacts must be inside the corpus root") from exc
        return {"file": relative, "sha256": _hash(path)}

    manifest = {
        "schemaVersion": 1, "frozenAt": datetime.now(timezone.utc).isoformat(),
        "split": labels.get("split"), "labels": entry(labels_path),
        "sources": [entry(path) for path in sources],
        "rules": [entry(normalized_file(str(path), root)) for path in rule_files],
        "protocol": "hash-current-artifacts; no claim of pre-development independence",
    }
    try:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise EvaluationError("refusing to overwrite an existing freeze") from exc
    return manifest


def verify_frozen(labels_path, manifest_path, root="."):
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or not manifest.get("frozenAt"):
        raise EvaluationError("invalid freeze manifest")
    labels = json.loads(Path(labels_path).read_text())
    cases = _cases(labels, root)
    try:
        if normalized_file(manifest["labels"]["file"], root) != normalized_file(str(labels_path)):
            raise EvaluationError("freeze identifies a different labels file")
        if manifest.get("split") != labels.get("split"):
            raise EvaluationError("frozen split does not match labels")
        entries = [manifest["labels"], *manifest["sources"], *manifest["rules"]]
        for entry in entries:
            if _hash(normalized_file(entry["file"], root)) != entry["sha256"]:
                raise EvaluationError(f"frozen artifact changed: {entry['file']}")
        frozen_sources = {normalized_file(entry["file"], root) for entry in manifest["sources"]}
        if frozen_sources != {case["file"] for case in cases}:
            raise EvaluationError("frozen source set does not match labelled files")
        if labels.get("split") == "holdout" and not manifest["rules"]:
            raise EvaluationError("holdout manifest is missing rule hashes")
    except (KeyError, TypeError) as exc:
        raise EvaluationError("malformed freeze manifest") from exc
    return manifest
