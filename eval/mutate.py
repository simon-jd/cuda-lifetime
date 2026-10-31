"""Reproducible, hash-pinned source mutations; never run the checker here."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re


class MutationError(ValueError):
    """Input source or mutation does not match the reviewed recipe."""


def digest(data):
    return sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def unique_replace(source, before, after):
    if not isinstance(before, str) or not before or not isinstance(after, str) or before == after:
        raise MutationError("replacement must contain distinct before/after text")
    count = source.count(before)
    if count != 1:
        raise MutationError(f"replacement must match exactly once, found {count}")
    return source.replace(before, after, 1)


def _inside(root, relative):
    if not isinstance(relative, str) or not relative:
        raise MutationError("source path must be a nonempty relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise MutationError("source path escapes the immutable source root")
    return path


def _write_identical_or_new(path, data):
    data = data.encode() if isinstance(data, str) else data
    if path.exists():
        if path.read_bytes() != data:
            raise MutationError(f"refusing to overwrite changed generated artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def generate(recipes, source_root, output, expected_manifest=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.is_relative_to(source_root):
        raise MutationError("output must be outside the immutable upstream source tree")
    if not isinstance(recipes, dict) or recipes.get("schemaVersion") != 1 or not isinstance(recipes.get("mutations"), list):
        raise MutationError("expected schemaVersion 1 recipes with mutations")
    prepared, baselines, names = [], {}, set()
    # Validate every input and every edit before creating any output artifacts.
    for recipe in recipes["mutations"]:
        if not isinstance(recipe, dict):
            raise MutationError("each mutation recipe must be an object")
        name = recipe.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise MutationError("mutation IDs must be unique lowercase identifiers")
        names.add(name)
        path = _inside(source_root, recipe.get("source"))
        original_bytes = path.read_bytes()
        original_hash = digest(original_bytes)
        if original_hash != recipe.get("sourceSha256"):
            raise MutationError(f"upstream source hash changed: {recipe['source']}")
        original = original_bytes.decode("utf-8")
        changed, edits = original, []
        if not isinstance(recipe.get("edits"), list) or not recipe["edits"]:
            raise MutationError("each mutation requires explicit edits")
        for edit in recipe["edits"]:
            if not isinstance(edit, dict):
                raise MutationError("each edit must be an object")
            before, after = edit.get("before"), edit.get("after")
            next_source = unique_replace(changed, before, after)
            edits.append({"before": before, "after": after, "matchCount": 1,
                          "inputLine": changed[:changed.index(before)].count("\n") + 1,
                          "inputSha256": digest(changed), "outputSha256": digest(next_source)})
            changed = next_source
        oracle = dict(recipe.get("oracle", {}))
        sites = []
        for site in oracle.get("sites", []):
            anchor = site["anchor"]
            if changed.count(anchor) != 1:
                raise MutationError(f"oracle source anchor is not unique: {name}")
            sites.append({**site, "line": changed[:changed.index(anchor)].count("\n") + 1})
        oracle["sites"] = sites
        baseline = "originals/" + path.name
        if baseline in baselines and baselines[baseline]["sha256"] != original_hash:
            raise MutationError("different upstream files would collide in baseline output")
        baselines[baseline] = {"source": recipe["source"], "file": baseline, "sha256": original_hash, "data": original_bytes}
        prepared.append({"id": name, "category": recipe["category"], "file": "generated/" + name + ".cu",
                         "source": recipe["source"], "sourceUrl": recipe["sourceUrl"],
                         "sourceSha256": original_hash, "parentSha256": original_hash,
                         "mutantSha256": digest(changed), "baseline": baseline,
                         "edits": edits, "oracle": oracle, "data": changed})
    if expected_manifest is not None:
        expected = {item["id"]: item["mutantSha256"] for item in expected_manifest["mutants"]}
        if expected != {item["id"]: item["mutantSha256"] for item in prepared}:
            raise MutationError("replayed output hashes differ from the recorded manifest")
    for baseline in baselines.values():
        _write_identical_or_new(output / baseline["file"], baseline["data"])
    for mutation in prepared:
        _write_identical_or_new(output / mutation["file"], mutation["data"])
    manifest = {"schemaVersion": 1, "upstream": recipes.get("upstream"),
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "oracleOrigin": "human-authored before any analysis of these generated mutants",
                "executionStatus": "not-run-at-generation", "recipes": recipes,
                "baselines": [{k: v for k, v in baseline.items() if k != "data"} for baseline in baselines.values()],
                "mutants": [{k: v for k, v in mutation.items() if k != "data"} for mutation in prepared]}
    # An existing manifest is retained if its generated source and recipe identity
    # match; reruns must not fabricate a new original generation timestamp.
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("recipes") != recipes or existing.get("mutants") != manifest["mutants"]:
            raise MutationError("refusing to overwrite a different mutation manifest")
        return existing
    _write_identical_or_new(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--recipes")
    origin.add_argument("--replay-manifest")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        expected = json.loads(Path(args.replay_manifest).read_text()) if args.replay_manifest else None
        recipes = expected["recipes"] if expected else json.loads(Path(args.recipes).read_text())
        manifest = generate(recipes, args.source_root, args.out, expected)
        print(json.dumps({"mutants": len(manifest["mutants"]), "baselines": len(manifest["baselines"]),
                          "manifest": str(Path(args.out).resolve() / "manifest.json")}, indent=2))
        return 0
    except (MutationError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"mutate: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
