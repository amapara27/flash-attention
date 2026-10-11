"""Profile every config that passed the correctness gate, then time it without the profiler.

For each PASS row in results/correctness.csv (workloads A0 / A / B):
  1. ncu, fixed metric set (metrics.txt), clocks locked to base, caches flushed,
     1 warmup launch skipped + 5 launches profiled  -> results/raw/<stage>/<name>.ncu-rep (+ .csv)
  2. bench --mode time (boost clocks, no profiler, median of 50)  -> results/timing.csv

Existing reports are reused unless --force. Needs GPU perf-counter access
(admin terminal, or NVIDIA Control Panel > Developer > Manage GPU Performance Counters > all users).

usage: python run_ncu.py [--force] [stage_name_prefix ...]      e.g. python run_ncu.py 06 07
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

from matrix import STAGES

ROOT = Path(__file__).parent
NCU = os.environ.get(
    "NCU", r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe")
RAW = ROOT / "results" / "raw"
TIMING = ROOT / "results" / "timing.csv"
STAGE_ID = {v: k for k, v in STAGES.items()}


def metrics():
    lines = (ROOT / "metrics.txt").read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def report_name(r):
    return f"{r['workload']}_{r['dtype']}_br{r['br']}_bc{r['bc']}"


def bench_args(r, mode):
    return [str(ROOT / f"bin/bench_s{STAGE_ID[r['stage']]}.exe"), "--fixture", f"fixtures/{r['fixture']}",
            "--causal", r["causal"], "--br", r["br"], "--bc", r["bc"], "--mode", mode]


def profile(r, force):
    out = RAW / r["stage"] / report_name(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep, csv_path = out.with_suffix(".ncu-rep"), out.with_suffix(".csv")
    if rep.exists() and csv_path.exists() and not force:
        return "cached"
    cmd = [NCU, "--clock-control", "base", "--cache-control", "all", "--replay-mode", "kernel",
           "-k", "regex:flash_attention", "--launch-skip", "1", "--launch-count", "5",
           "--metrics", ",".join(metrics()), "--section", "SourceCounters",
           "-f", "-o", str(out)] + bench_args(r, "profile")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if "ERR_NVGPUCTRPERM" in p.stdout + p.stderr:
        sys.exit("ncu: no permission for GPU performance counters - see the docstring")
    if not rep.exists():
        return f"NCU FAILED: {(p.stdout + p.stderr)[-300:]}"
    with open(csv_path, "w") as f:
        subprocess.run([NCU, "--import", str(rep), "--csv", "--page", "raw"], stdout=f, text=True)
    return "ok"


def wall_time(r):
    p = subprocess.run(bench_args(r, "time"), cwd=ROOT, capture_output=True, text=True)
    fields = p.stdout.strip().splitlines()[-1].split(",")
    return fields[-1] if fields[10] == "PASS" else ""


def main():
    force = "--force" in sys.argv
    prefixes = [a for a in sys.argv[1:] if a != "--force"]
    with open(ROOT / "results" / "correctness.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["status"] == "PASS" and r["workload"] in ("A0", "A", "B")
                and (not prefixes or any(r["stage"].startswith(p) for p in prefixes))]

    timings = {}
    if TIMING.exists():
        with open(TIMING, newline="") as f:
            timings = {(t["stage"], t["name"]): t["wall_ms"] for t in csv.DictReader(f)}

    for i, r in enumerate(rows, 1):
        status = profile(r, force)
        wall = wall_time(r)
        timings[(r["stage"], report_name(r))] = wall
        print(f"[{i:3d}/{len(rows)}] {r['stage']:22s} {report_name(r):24s} ncu={status:7s} wall={wall} ms", flush=True)

    with open(TIMING, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "name", "wall_ms"])
        for (stage, name), ms in sorted(timings.items()):
            w.writerow([stage, name, ms])


if __name__ == "__main__":
    main()
