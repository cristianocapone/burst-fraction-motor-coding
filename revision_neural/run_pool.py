"""Run shell commands from a jobs file with at most P in parallel; one log per job.

Usage: python run_pool.py jobs.txt P logdir [wait_for_file]
Each line of jobs.txt: <logname> <command...>. If wait_for_file is given, the
pool starts only when that file exists (e.g. a download DONE marker)."""
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

jobs_file, P, logdir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
wait_for = Path(sys.argv[4]) if len(sys.argv) > 4 else None
logdir.mkdir(exist_ok=True)
while wait_for is not None and not wait_for.exists():
    time.sleep(60)
jobs = [l.split(maxsplit=1) for l in Path(jobs_file).read_text().splitlines() if l.strip() and not l.startswith("#")]


def run(job):
    name, cmd = job
    t0 = time.time()
    with open(logdir / f"{name}.log", "a") as log:
        log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd}\n"); log.flush()
        rc = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{name}: rc={rc} {time.time()-t0:.0f}s", flush=True)
    return rc


with ThreadPoolExecutor(P) as ex:
    rcs = list(ex.map(run, jobs))
(logdir / "DONE").write_text(f"{time.strftime('%Y-%m-%d %H:%M')} failed={sum(r != 0 for r in rcs)}/{len(rcs)}\n")
print("all done, failed:", sum(r != 0 for r in rcs), flush=True)
