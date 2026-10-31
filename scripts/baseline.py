#!/usr/bin/env python3
"""Run exact compiler baselines without importing checker code or relabeling cases."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frontend_args(entry):
    args = entry.get("arguments") or shlex.split(entry["command"])
    output = []
    skip = False
    for arg in args[1:]:
        if skip:
            skip = False
        elif arg in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif arg in ("-c", "-MD", "-MMD", "-MP"):
            pass
        else:
            output.append(arg)
    return output


def run_one(index, entry, args, input_command_sha256):
    source = str((Path(entry["directory"]) / entry["file"]).resolve())
    key = f"{index:04d}-" + hashlib.sha256(source.encode()).hexdigest()[:10]
    log = args.out / (key + ".log")
    artifact = args.out / (key + (".yaml" if args.mode == "tidy" else ".sarif"))
    inputs = args.out / "inputs" / key
    inputs.mkdir(parents=True)
    isolated_entry = {**entry, "file": source}
    isolated_database = inputs / "compile_commands.json"
    isolated_database.write_text(json.dumps([isolated_entry], indent=2) + "\n")
    if args.mode == "tidy":
        command = [args.tidy, source, "-p", str(inputs),
                   "--checks=-*,clang-analyzer-*,bugprone-*", "--export-fixes=" + str(artifact)]
    else:
        command = [args.clang, *frontend_args(entry)]
        command += ["-fsyntax-only"] if args.mode == "syntax" else [
            "--analyze", "-Xclang", "-analyzer-output=sarif", "-o", str(artifact),
            "-Xclang", "-analyzer-config", "-Xclang", f"max-nodes={args.max_nodes}"]
    rss = args.out / (key + ".rss")
    timed = ["/usr/bin/time", "-f", "%M", "-o", str(rss), "--", *command] if Path('/usr/bin/time').is_file() else command
    started = time.monotonic()
    record = {"file":source,"command":command,"mode":args.mode,"log":str(log),
              "artifact":str(artifact),
              "sourceSha256":sha256(Path(source)) if Path(source).is_file() else None,
              "inputDatabase":str(isolated_database),
              "inputDatabaseSha256":hashlib.sha256(isolated_database.read_bytes()).hexdigest(),
              "inputCommandSha256":input_command_sha256,
              "workingDirectory":entry['directory'], "compilationCommandIndex":index}
    try:
        with log.open('w') as stream:
            proc = subprocess.Popen(timed, cwd=entry['directory'], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record['returncode'] = proc.wait(timeout=args.timeout)
                record['status'] = 'passed' if record['returncode'] == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                record['status'] = 'timeout'
    except OSError as exc:
        record.update(status='unavailable', error=str(exc))
    record['seconds'] = time.monotonic() - started
    record['sourceUnchanged'] = (record['sourceSha256'] is not None and Path(source).is_file()
                                 and sha256(Path(source)) == record['sourceSha256'])
    record['inputDatabaseUnchanged'] = (isolated_database.is_file()
                                        and sha256(isolated_database) == record['inputDatabaseSha256'])
    if rss.exists():
        values = [line for line in rss.read_text().splitlines() if line.isdecimal()]
        record['peakRssKiB'] = int(values[-1]) if values else None
    if log.exists():
        record['diagnosticLines'] = [line for line in log.read_text(errors='replace').splitlines() if ': warning:' in line or ': error:' in line]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compdb', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['syntax','analyze','tidy'], required=True)
    parser.add_argument('--clang', default='clang++')
    parser.add_argument('--tidy', default='clang-tidy')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--max-nodes', type=int, default=225000)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if min(args.timeout, args.max_nodes, args.jobs) <= 0:
        parser.error('budgets must be positive')
    if args.compdb.is_dir(): args.compdb /= 'compile_commands.json'
    args.compdb = args.compdb.resolve(); args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error('use an empty output directory to preserve prior baseline evidence')
    for name in ('clang','tidy'):
        exe = getattr(args,name)
        setattr(args,name,str(Path(exe).resolve()) if Path(exe).is_file() else shutil.which(exe) or exe)
    database_bytes = args.compdb.read_bytes()
    (args.out / 'original-compile_commands.json').write_bytes(database_bytes)
    entries = json.loads(database_bytes)
    if not isinstance(entries, list) or not entries:
        parser.error('compile database must contain at least one command')
    command_hashes = [hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
                      for entry in entries]
    for entry in entries:
        entry['directory'] = str((args.compdb.parent / entry['directory']).resolve())
    executable = args.tidy if args.mode == 'tidy' else args.clang
    executable_hash = sha256(Path(executable)) if Path(executable).is_file() else None
    try:
        version = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=30)
        tool_version = {'executable':executable, 'returncode':version.returncode,
                        'stdout':version.stdout, 'stderr':version.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        tool_version = {'executable':executable, 'error':str(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda pair: run_one(*pair, args, command_hashes[pair[0]]), enumerate(entries)))
    result = {'schemaVersion':1,'mode':args.mode,'compdb':str(args.compdb),
              'compdbSha256':hashlib.sha256(database_bytes).hexdigest(),
              'compdbUnchanged':args.compdb.is_file() and args.compdb.read_bytes() == database_bytes,
              'toolVersion':tool_version,'executableSha256':executable_hash,
              'executableUnchanged':executable_hash is not None and Path(executable).is_file() and sha256(Path(executable)) == executable_hash,
              'jobs':args.jobs,'records':records,
              'externalInputs':'per-command databases and TU hashes are retained; compiler SDKs, libraries and transitive headers remain external',
              'summary':{status:sum(r['status']==status for r in records) for status in ('passed','failed','timeout','unavailable')},
              'warningCount':sum(sum(': warning:' in line for line in r.get('diagnosticLines',[])) for r in records)}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']))
    return 0 if result['compdbUnchanged'] and result['executableUnchanged'] and all(
        r['status']=='passed' and r['sourceUnchanged'] and r['inputDatabaseUnchanged'] for r in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
