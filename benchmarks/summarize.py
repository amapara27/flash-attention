"""Build results/summary.csv: one row per stage x workload x dtype x tile, metrics as columns.

Sources: results/correctness.csv (every config, incl. FAIL / not profiled),
         results/raw/<stage>/*.csv (ncu, median over the 5 profiled launches),
         results/timing.csv (boost-clock wall time).
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
RES = ROOT / "results"

# ncu metric -> summary column
SHORT = {
    "gpu__time_duration.sum": "ncu_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_sol_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_sol_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "dram__bytes_read.sum": "dram_read_bytes",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "l1_sol_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_sol_pct",
    "dram__bytes_write.sum": "dram_write_bytes",
    "lts__t_bytes.sum": "l2_bytes",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global_ld_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "global_ld_requests",
    "smsp__inst_executed_op_ldgsts.sum": "cp_async_inst",
    "sm__sass_data_bytes_mem_global_op_ldgsts.sum": "cp_async_bytes",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occ_achieved_pct",
    "sm__maximum_warps_per_active_cycle_pct": "occ_theoretical_pct",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
    "launch__registers_per_thread": "regs_per_thread",
    "launch__shared_mem_per_block_static": "smem_static_bytes",
    "launch__occupancy_limit_registers": "occ_limit_regs",
    "launch__occupancy_limit_shared_mem": "occ_limit_smem",
    "launch__occupancy_limit_warps": "occ_limit_warps",
    "launch__occupancy_limit_blocks": "occ_limit_blocks",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "smem_conflicts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum": "smem_wavefronts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "smem_ld_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "smem_st_conflicts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "smem_ld_wavefronts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "smem_st_wavefronts",
    "smsp__inst_executed_op_shared_ld.sum": "smem_ld_inst",
    "smsp__inst_executed_op_shared_st.sum": "smem_st_inst",
    "smsp__inst_executed_op_ldsm.sum": "ldmatrix_inst",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum": "local_ld_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum": "local_st_bytes",
    "smsp__inst_executed.sum": "inst_executed",
    "sm__inst_executed.avg.per_cycle_active": "ipc",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_slot_pct",
    "smsp__warps_active.avg.per_cycle_active": "warps_active_per_sched",
    "smsp__warps_eligible.avg.per_cycle_active": "warps_eligible_per_sched",
    "smsp__thread_inst_executed_per_inst_executed.ratio": "threads_per_inst",
    "smsp__sass_branch_targets_threads_divergent.sum": "divergent_branches",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "fma_pipe_pct",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": "tensor_pipe_pct",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active": "xu_pipe_pct",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma_thread_inst",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": "hfma_thread_inst",
    "smsp__inst_executed_pipe_tensor_op_hmma.sum": "hmma_inst",
}
# stall reasons, in cycles per issued instruction; they sum to warp_latency_cycles
STALLS = ("long_scoreboard", "barrier", "short_scoreboard", "mio_throttle", "math_pipe_throttle",
          "wait", "not_selected", "lg_throttle", "selected", "branch_resolving", "dispatch_stall",
          "drain", "imc_miss", "membar", "misc", "no_instruction", "sleeping", "tex_throttle")
SHORT["smsp__average_warp_latency_per_inst_issued.ratio"] = "warp_latency_cycles"
for s in STALLS:
    SHORT[f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio"] = f"stall_{s}"


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


# ncu auto-scales units per report (ms vs us, Kbyte vs Mbyte); normalise to ns and bytes.
# byte prefixes are decimal: 134.217728 Mbyte == 134,217,728 B (verified on a known cp.async count)
UNIT_SCALE = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9, "msecond": 1e6, "usecond": 1e3, "nsecond": 1,
              "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}


def read_ncu(path):
    """median of each metric over the profiled launches, in base units"""
    lines = [l for l in path.read_text().splitlines() if l.startswith('"')]
    rows = list(csv.reader(lines))
    if len(rows) < 3:
        return {}
    header, units, data = rows[0], rows[1], rows[2:]
    out = {}
    for metric, col in SHORT.items():
        if metric in header:
            i = header.index(metric)
            scale = UNIT_SCALE.get(units[i], 1)
            vals = [v * scale for v in (num(r[i]) for r in data) if v is not None]
            if vals:
                out[col] = statistics.median(vals)
    return out


def shape(fixture):
    B, N, H, DK, DV = map(int, (ROOT / "fixtures" / fixture / "shape.txt").read_text().split())
    return B, N, H, DK, DV


def main():
    with open(RES / "correctness.csv", newline="") as f:
        gate = [r for r in csv.DictReader(f) if r["workload"] in ("A0", "A", "B")]
    timing = {}
    if (RES / "timing.csv").exists():
        with open(RES / "timing.csv", newline="") as f:
            timing = {(t["stage"], t["name"]): t["wall_ms"] for t in csv.DictReader(f)}

    base = ["stage", "workload", "fixture", "causal", "dtype", "br", "bc", "threads", "smem_bytes",
            "status", "max_err", "max_err_vs_fp32", "wall_ms", "ncu_ms", "tflops_wall", "tflops_ncu",
            "smem_conflict_pct", "smem_ld_conflict_pct", "global_ld_sectors_per_req"]
    out = []
    for r in gate:
        name = f"{r['workload']}_{r['dtype']}_br{r['br']}_bc{r['bc']}"
        row = {k: r.get(k, "") for k in base}
        p = RES / "raw" / r["stage"] / f"{name}.csv"
        m = read_ncu(p) if p.exists() else {}
        row.update(m)
        row["wall_ms"] = timing.get((r["stage"], name), "")

        B, N, H, DK, DV = shape(r["fixture"])
        flops = 2 * B * H * N * N * (DK + DV) * (0.5 if r["causal"] == "1" else 1.0)   # useful FLOPs
        if m.get("ncu_ns"):
            row["ncu_ms"] = m["ncu_ns"] / 1e6
            row["tflops_ncu"] = flops / (m["ncu_ns"] * 1e-9) / 1e12
        if num(row["wall_ms"]):
            row["tflops_wall"] = flops / (num(row["wall_ms"]) * 1e-3) / 1e12
        if m.get("smem_ld_wavefronts"):
            row["smem_ld_conflict_pct"] = 100 * m.get("smem_ld_conflicts", 0) / m["smem_ld_wavefronts"]
        if m.get("smem_wavefronts"):
            row["smem_conflict_pct"] = 100 * m.get("smem_conflicts", 0) / m["smem_wavefronts"]
        if m.get("global_ld_requests"):
            row["global_ld_sectors_per_req"] = m["global_ld_sectors"] / m["global_ld_requests"]
        out.append(row)

    cols = base + [c for c in SHORT.values() if c not in base]
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"{len(out)} rows -> {RES / 'summary.csv'}")


if __name__ == "__main__":
    main()
