"""Public command line. Analysis always requires the native CSA executable."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cuda-lifetime")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("translate")
    tr.add_argument("--compdb", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--cuda-path")
    tr.add_argument("--clang", default="clang++")
    tr.add_argument("--gcc-toolchain", help="Preserve the nvcc host compiler's GCC header/toolchain prefix")
    tr.add_argument("--compatibility-profile", help="Explicit header profile template; creates a resolved input/tool/header lock")
    ma = sub.add_parser("macros")
    ma.add_argument("--compdb", required=True)
    ma.add_argument("--out", required=True)
    ma.add_argument("--timeout", type=float, default=30)
    an = sub.add_parser("analyze")
    an.add_argument("--compdb", required=True)
    an.add_argument("--model", default="models")
    an.add_argument("--sarif", required=True)
    an.add_argument("--coverage")
    an.add_argument("--native", default="build/cuda-lifetime-native")
    an.add_argument("--max-nodes", type=int, default=225000)
    an.add_argument("--timeout", type=float, default=180)
    an.add_argument("--errors-only", action="store_true",
                    help="hide warning-level diagnostics, which depend on an assumed contract or unknown effects")
    an.add_argument("--include-contract", action="store_true", help=argparse.SUPPRESS)  # the default since 0.3.0
    ev = sub.add_parser("eval")
    ev.add_argument("args", nargs=argparse.REMAINDER)
    # Delegate eval's arguments directly, retaining its independent package CLI.
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "eval":
        from eval.__main__ import main as eval_main
        return eval_main(args_list[1:])
    args = parser.parse_args(args_list)
    try:
        if args.command == "translate":
            from translate import translate_compdb
            report = translate_compdb(args.compdb, args.out, args.cuda_path, args.clang,
                                     gcc_toolchain=args.gcc_toolchain, compatibility_profile=args.compatibility_profile)
            print(json.dumps(report.get("summary", report), ensure_ascii=False))
            return 0 if not any(x.get("status") == "translation-unknown" for x in report.get("entries", [])) else 2
        if args.command == "macros":
            from translate import macro_diff_compdb
            report = macro_diff_compdb(args.compdb, args.out, timeout=args.timeout)
            print(json.dumps(report["summary"], ensure_ascii=False))
            return 0 if all(entry["status"] == "compared" and entry["requiredDifferences"]["equal"]
                            and entry.get("profileHeaderEvidence", {}).get("status", "not-applicable") in ("verified", "not-applicable")
                            for entry in report["entries"]) else 2
        from tool.runner import analyze
        coverage = analyze(args.compdb, args.sarif, args.coverage, args.native,
                           args.model, args.max_nodes, args.timeout, not args.errors_only)
        print(json.dumps(coverage["summary"], ensure_ascii=False))
        return 0 if coverage["complete"] else 2
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"cuda-lifetime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
