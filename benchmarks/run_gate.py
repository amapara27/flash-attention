"""Correctness gate: run every (stage, workload, tile) un-profiled and record PASS / FAIL.

Each config is checked REPEATS times (more for the wmma stages, which have a suspected race);
a config passes only if every repeat passes. Writes results/correctness.csv.
Also runs the stage-8 ragged-shape checks (large_odd, batch_odd, causal and non-causal).

usage: python run_gate.py [stage ...]
"""
import csv
import subprocess
import sys
from pathlib import Path

from matrix import STAGES, WORKLOADS, STAGE_WORKLOADS, TILES, ODD_FIXTURES, binary

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "correctness.csv"
FIELDS = ["stage", "fixture", "causal", "dtype", "br", "bc", "threads", "smem_bytes",
          "max_err", "max_err_vs_fp32", "status", "wall_ms"]


def run_check(exe, fixture, causal, br, bc):
    cmd = [str(ROOT / exe), "--fixture", f"fixtures/{fixture}", "--causal", str(causal),
           "--br", str(br), "--bc", str(bc), "--mode", "check"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    row = dict(zip(FIELDS, line.split(",")))
    if not row.get("status"):
        row = {"status": f"CRASH:{p.returncode}"}
    return row


def gate(exe, stage_name, fixture, causal, br, bc, repeats, workload):
    worst = None
    for _ in range(repeats):
        row = run_check(exe, fixture, causal, br, bc)
        if row["status"] == "UNSUPPORTED":
            return None
        if worst is None or row["status"] != "PASS" or \
                float(row.get("max_err") or "inf") > float(worst.get("max_err") or 0):
            worst = row
        if row["status"] != "PASS":
            break
    worst.update(stage=stage_name, fixture=fixture, causal=str(causal), br=str(br), bc=str(bc))
    worst["workload"] = workload
    worst["repeats"] = str(repeats)
    return worst


def main():
    stages = [int(s) for s in sys.argv[1:]] or list(STAGES)
    rows = []
    for s in stages:
        repeats = 10 if s in (9, 91) else 3
        for wl in STAGE_WORKLOADS[s]:
            fixture, causal = WORKLOADS[wl]
            for br, bc in TILES:
                r = gate(binary(s), STAGES[s], fixture, causal, br, bc, repeats, wl)
                if r:
                    rows.append(r)
                    print(f"{STAGES[s]:22s} {wl:3s} {br:3d}x{bc:<3d} {r['status']:10s} "
                          f"err={r.get('max_err', '')} vs_fp32={r.get('max_err_vs_fp32', '')}", flush=True)
        if s == 8:
            for fixture in ODD_FIXTURES:
                for causal in (0, 1):
                    for br, bc in TILES:
                        r = gate("bin/bench_s8_odd.exe", STAGES[8], fixture, causal, br, bc, 3, "odd")
                        if r:
                            rows.append(r)
                            print(f"{STAGES[8]:22s} {fixture}/c{causal} {br:3d}x{bc:<3d} {r['status']:10s} "
                                  f"err={r.get('max_err', '')} vs_fp32={r.get('max_err_vs_fp32', '')}", flush=True)

    # merge with rows for stages not re-run
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if OUT.exists():
        with open(OUT, newline="") as f:
            keep = [r for r in csv.DictReader(f) if r["stage"] not in {STAGES[s] for s in stages}]
    fields = ["workload", "repeats"] + FIELDS
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep + rows)

    bad = [r for r in rows if r["status"] != "PASS"]
    print(f"\n{len(rows)} configs checked, {len(bad)} not passing -> {OUT}")


if __name__ == "__main__":
    main()
