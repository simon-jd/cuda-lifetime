"""Standalone translator and macro evidence CLI."""

import argparse
import json
import sys

from . import TranslationError, macro_diff_compdb, translate_compdb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    translate = sub.add_parser('translate', help='translate a compilation database')
    translate.add_argument('--compdb', required=True)
    translate.add_argument('--out', required=True)
    translate.add_argument('--cuda-path')
    translate.add_argument('--clang', default='clang++')
    translate.add_argument('--gcc-toolchain', help='GCC installation prefix used for host standard-library headers')
    translate.add_argument('--compatibility-profile', help='Explicit compatibility template; automatically resolves current input identities')
    macros = sub.add_parser('macros', help='compare real preprocessor macro sets')
    macros.add_argument('--compdb', required=True, help='translated directory or translation-report.json')
    macros.add_argument('--out')
    macros.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == 'translate':
            report = translate_compdb(args.compdb, args.out, args.cuda_path, args.clang, args.gcc_toolchain,
                                      compatibility_profile=args.compatibility_profile)
            status = 2 if report['summary']['translationUnknown'] else 0
        else:
            report = macro_diff_compdb(args.compdb, args.out, args.timeout)
            status = 2 if any(e['status'] != 'compared' or not e['requiredDifferences']['equal'] or
                             e.get('profileHeaderEvidence', {}).get('status', 'not-applicable') not in ('verified', 'not-applicable')
                             for e in report['entries']) else 0
    except (TranslationError, OSError) as error:
        parser.exit(1, f'error: {error}\n')
    print(json.dumps(report['summary'], indent=2))
    return status


if __name__ == '__main__':
    sys.exit(main())
