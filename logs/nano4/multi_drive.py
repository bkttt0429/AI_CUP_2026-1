#!/usr/bin/env python3
"""Driver of a multi-node job (scripts/nano4/run_multi.sbatch, 2026-11-17): runs on the job's first node and starts every task
as a FULL-NODE srun step (8 GPUs, 96 cores) on one of the job's nodes. Tasks are files in the spool folder and can be added
while the job runs:
  <spool>/<name>.task      line 1: node index (0 .. N-1) or "any"; the rest: the bash command (run from the project root)
The driver renames a started task to <name>.run.<node> and a finished one to <name>.done.<exit code>; the step's output goes to
logs/nano4/<name>-<jobid>.out. A node runs one step at a time. <spool>/STOP: start nothing new and leave when the running
steps have ended. The driver also leaves when nothing has run and nothing has waited for --idle-exit minutes (the job then ends
and gives its nodes back).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spool", required=True); ap.add_argument("--idle-exit", type=float, default=45.0); a = ap.parse_args()
    spool = Path(a.spool); job = os.environ["SLURM_JOB_ID"]; nodes = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], capture_output=True, text=True).stdout.split()
    log(f"driver on {os.uname().nodename}; job {job}; {len(nodes)} nodes: {' '.join(nodes)}; spool {spool}")
    busy = {}; idle_since = time.time()  # node -> (name, process, output file, running-marker path)
    while True:
        for node, (name, p, fh, mark) in list(busy.items()):
            rc = p.poll()
            if rc is not None:
                fh.close(); mark.rename(spool / f"{name}.done.{rc}"); del busy[node]; log(f"task {name} on {node} ended with exit code {rc}")
        stop = (spool / "STOP").exists()
        if not stop:
            for f in sorted(spool.glob("*.task")):
                lines = f.read_text().splitlines()
                if len(lines) < 2:
                    continue  # still being written
                want = lines[0].strip(); cmd = "\n".join(lines[1:]); free = [n for n in nodes if n not in busy]
                node = (free[0] if free else None) if want == "any" else (nodes[int(want)] if nodes[int(want)] not in busy else None)
                if node is None:
                    continue
                name = f.name[:-5]; mark = spool / f"{name}.run.{node}"; f.rename(mark); fh = open(f"logs/nano4/{name}-{job}.out", "a")
                p = subprocess.Popen(["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=96", "--gres=gpu:8", f"--nodelist={node}", f"--job-name={name}", "bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT)
                busy[node] = (name, p, fh, mark); log(f"task {name} started on {node}: {cmd[:300]}")
        waiting = any(spool.glob("*.task"))
        if busy or (waiting and not stop):
            idle_since = time.time()
        if not busy and (stop or time.time() - idle_since > a.idle_exit * 60):
            log("nothing running" + (" (STOP)" if stop else f" for {a.idle_exit:.0f} minutes") + ": leaving"); return 0
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
