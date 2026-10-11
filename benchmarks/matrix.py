"""Benchmark matrix: which stages run which workloads at which tile sizes."""

STAGES = {   # stage id (STAGE macro) -> stage name
    1: "01_tiled",
    2: "02_causal_mask",
    3: "03_tile_skipping",
    4: "04_multihead",
    5: "05_s_registers",
    6: "06_warp_partitioning",
    7: "07_fp16_smem",
    8: "08_double_buffering",
    9: "09_wmma_qk",
    91: "09b_wmma_qk_fixed",
}

WORKLOADS = {   # id -> (fixture, causal)
    "A0": ("single_4096", 0),   # B=1 H=1 N=4096, non-causal
    "A":  ("single_4096", 1),   # B=1 H=1 N=4096, causal
    "B":  ("batch4_4096", 1),   # B=1 H=4 N=4096, causal
}

STAGE_WORKLOADS = {
    1: ["A0"],
    2: ["A0", "A"],
    3: ["A"],
    **{s: ["A", "B"] for s in (4, 5, 6, 7, 8, 9, 91)},
}

# every stage is offered the full grid; the harness reports UNSUPPORTED for configs over 48 KB smem
# (or B_c != 32 for the committed wmma kernel)
TILES = [(br, bc) for br in (16, 32, 64, 128) for bc in (16, 32, 64)]

# ragged-shape correctness checks (bench_s8_odd), not profiled
ODD_FIXTURES = ["large_odd", "batch_odd"]


def binary(stage):
    return f"bin/bench_s{stage}.exe"
