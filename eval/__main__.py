"""python -m eval: evaluate SARIF or record a write-once corpus freeze."""

import argparse
import json
from pathlib import Path
import sys

from . import EvaluationError, create_freeze, evaluate, verify_frozen


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    freeze_mode = bool(argv and argv[0] == "freeze")
    if freeze_mode:
        argv.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", required=freeze_mode)
    if freeze_mode:
        parser.add_argument("--rule-file", action="append", default=[])
    else:
        parser.add_argument("--sarif", required=True)
        parser.add_argument("--coverage", required=True)
        parser.add_argument("--freeze", help="verify hashes before evaluating (required for holdout)")
    args = parser.parse_args(argv)
    try:
        if freeze_mode:
            result = create_freeze(args.labels, args.out, args.root, args.rule_file)
        else:
            labels = json.loads(Path(args.labels).read_text())
            if labels.get("split") == "holdout" and not args.freeze:
                raise EvaluationError("holdout evaluation requires --freeze")
            if args.freeze:
                verify_frozen(args.labels, args.freeze, args.root)
            result = evaluate(labels, json.loads(Path(args.sarif).read_text()), json.loads(Path(args.coverage).read_text()), args.root)
            if args.freeze:
                result["freezeVerified"] = True
            if args.out:
                Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if freeze_mode or result["passed"] else 1
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
