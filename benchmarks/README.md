# Benchmarks

Every optimisation stage of the kernel, re-extracted from the commit that finished it and benchmarked with one harness,
one Nsight Compute metric set, and one methodology. Results, per-stage write-ups and full tile sweeps: [`RESULTS.md`](RESULTS.md).

## Stages

| Stage | Commit | What it adds |
|---|---|---|
| `01_tiled` | `c4ad580` | multi-block, multi-tile FA2 with online softmax (non-causal) |
| `02_causal_mask` | `7de13d3` | per-element causal mask |
| `03_tile_skipping` | `81bc58e` | skip fully-masked K/V tiles |
| `04_multihead` | `3249aeb` | batched multi-head grid |
| `05_s_registers` | `b890554` | `S` in registers (compile-time loops) |
| `06_warp_partitioning` | `bde48e3` | two threads per row + shuffle reduction |
| `07_fp16_smem` | `5687ebd` | fp16 Q/K/V storage, fp32 accumulation |
| `08_double_buffering` | `c2d1ecf` | `cp.async` double-buffered K/V |
| `09_wmma_qk` | `5959273` | WMMA tensor cores for `QKᵀ` (has a race; only 1-warp tiles are correct) |
| `09b_wmma_qk_fixed` | `9f657c5` | stage 9 with the race and accumulator-reset fixes |

## Methodology

- **Correctness gate first.** Every stage × workload × tile is checked against fp64 CPU references (`make_refs.cu`) before it's
  profiled. fp16 stages are compared against a reference computed from the *same fp16-rounded inputs* (tol 1e-4), so kernel
  bugs aren't hidden by storage-rounding error. Configs that fail are recorded and not profiled.
- **Profiling:** Nsight Compute with clocks locked to base, caches flushed, 1 warmup launch skipped, median of 5 profiled launches,
  the same 69 metrics (`metrics.txt`) for every config, plus per-line source counters.
- **Wall time:** cudaEvent at boost clocks, median of 50 launches, no profiler.
- **Workloads:** A0 = 1 head, N = 4096, d = 64, non-causal · A = same, causal · B = 4 heads, causal.
- **Tiles:** B_r ∈ {16, 32, 64, 128} × B_c ∈ {16, 32, 64}, minus configs over 48 KB of static shared memory.

## Reproducing

Requirements: an sm_86 GPU, CUDA 12.x, MSVC (VS 2022) for `nvcc` on Windows, Nsight Compute, Python 3 (standard library only
for these scripts; PyTorch for the fixture step), and Git Bash for the `.sh` scripts.

```bash
python reference/dump_trace.py         # 1. PyTorch fixtures -> ../matrices
bash benchmarks/make_fixtures.sh       # 2. copy inputs + fp64 references -> benchmarks/fixtures
bash benchmarks/extract_stages.sh      # 3. each stage's kernel from git -> benchmarks/stages/*/kernel.cuh
bash benchmarks/build_all.sh           # 4. one binary per stage (parallel; ~10 min)
python benchmarks/run_gate.py          # 5. correctness gate -> results/correctness.csv
python benchmarks/run_ncu.py           # 6. profile + time every passing config -> results/raw/, results/timing.csv
python benchmarks/summarize.py         # 7. -> results/summary.csv
```

- `nvcc.bat` assumes VS 2022 Professional; set `VCVARS` to your `vcvars64.bat` otherwise. `run_ncu.py` reads `NCU` for the
  path to `ncu.exe`.
- Step 6 needs access to GPU performance counters: run it from an administrator terminal, or enable *NVIDIA Control Panel →
  Developer → Manage GPU Performance Counters → Allow access to all users*. Existing reports are reused unless `--force`.
- `sass_stats.py <stage> [name filter]` prints static SASS instruction counts per kernel instantiation (`LDS` widths,
  `LDL`/`STL`, `FFMA`, `HMMA`, …). It was used to confirm the register-allocation findings in stages 2 and 5.

## Layout

```
benchmarks/
├── bench.cu            # harness: one binary per stage (-DSTAGE=n), modes check / profile / time
├── make_refs.cu        # fp64 CPU attention references (fp32 and fp16-rounded inputs, causal and non-causal)
├── extract_stages.sh   # stage kernels from git history
├── make_fixtures.sh    # fixtures from ../matrices + make_refs
├── build_all.sh        # parallel build of every stage
├── nvcc.bat            # loads MSVC so nvcc finds cl.exe
├── matrix.py           # which stages run which workloads and tiles
├── metrics.txt         # the Nsight Compute metric set
├── run_gate.py         # correctness gate
├── run_ncu.py          # profiling + wall-time
├── summarize.py        # ncu CSVs -> results/summary.csv
├── sass_stats.py       # static SASS counts
├── RESULTS.md          # progression, findings, per-stage write-ups, tile sweeps
└── results/            # correctness.csv, summary.csv, timing.csv
```

Generated and not committed: `bin/`, `fixtures/`, `stages/*/kernel.cuh`, `results/raw/`.
