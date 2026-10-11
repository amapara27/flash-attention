# FlashAttention 2: Forward Kernel from Scratch

![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-reference%20%2B%20verification-EE4C2C?logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RTX%203060%20Ti%20(Ampere%2C%20CC%208.6)-blue)

A from-scratch CUDA implementation of the FlashAttention 2 forward pass for **prefill**, built for consumer Ampere. Prefill means attention over a full sequence at once: all N query rows against all N keys, as when a model processes a prompt. That's the case FA2's tiling targets. Single-token decode (one query row against a KV cache) is a different, latency-bound problem and isn't covered here. No cuBLAS, no cuDNN, no PyTorch in the kernel layer. The kernel is hand-written and verified element-wise against references.

The repo is the full derivation chain, not just the final kernel: a plain attention reference → a tiled online-softmax reference in PyTorch (with a per-iteration trace) → the fused CUDA kernel that implements the same recurrence on-chip → an optimisation ladder where every rung is re-verified and profiled with Nsight Compute.

## Current Status

**FA2 prefill forward is implemented, verified, and profiled at every optimisation stage.** Batched multi-head causal prefill at N = 4096, d = 64 runs in **2.66 ms on 4 heads (4.6 TFLOPS at boost clocks)**. That's **6.8× faster** than the first multi-head version and **10.7× faster** than the first causal kernel on a single head.

Next up: tensor cores via `mma.sync` PTX with S kept in registers. The profiling below says that's where the remaining time is.

## Results

Every stage was re-extracted from its commit and benchmarked with one harness, one Nsight Compute metric set (69 metrics), and one methodology, across a B_r × B_c ∈ {16, 32, 64, 128} × {16, 32, 64} tile sweep. Every configuration is checked against an fp64 reference before it's profiled.

**Best tile per stage, causal, N = 4096, d = 64** (Nsight Compute duration, clocks locked to base; boost-clock wall time in brackets):

| # | Stage | 1 head | 4 heads | Speedup (4 heads) |
|---|---|---|---|---|
| 2 | Causal masking | 8.82 ms (6.53) | — | — |
| 3 | + tile skipping | 7.39 ms (5.63) | — | — |
| 4 | + batched multi-head | 7.64 ms (5.80) | 18.20 ms (13.14) | baseline |
| 5 | `S` in registers | 3.05 ms (2.46) | 4.55 ms (3.40) | **4.00×** |
| 6 | Warp partitioning | 2.70 ms (2.11) | 5.26 ms (3.99) | 0.86× |
| 7 | fp16 storage | 2.16 ms (1.37) | 3.80 ms (2.86) | 1.38× |
| 8 | `cp.async` double buffering | 1.02 ms (0.77) | 2.74 ms (2.12) | **1.39×** |
| 9 | WMMA tensor cores for `QKᵀ` | **0.82 ms (0.58)** | **2.66 ms (1.88)** | 1.03× |

Best tiles end up small: 32×16 (1 head) and 64×16 (4 heads) for stage 8. Stage 1 (non-causal only) runs 3.42 ms (2.68).

### What the profiling showed

- **It's never memory-bound.** Peak DRAM utilisation across all 165 profiled configurations is **2.8%**. FlashAttention removes the HBM bottleneck by design, so every bottleneck here is on-chip: shared-memory bank conflicts, latency, and issue rate.
- **Turning on causal masking made the kernel 2.6× slower, but not because of the mask.** The causal instantiation got a different register allocation (94 vs 156 registers), so nvcc stopped keeping each thread's Q row in registers. It re-read Q from shared memory for every key column, with 8-way bank conflicts. That added 512 shared loads per warp per tile, and shared-memory stalls went from 0.5 to 8.3 cycles per instruction.
- **Tile skipping halved the work and saved 0% at 128×32.** That's 32 blocks on 38 SMs, and the last block needs every K/V tile regardless. The kernel finishes when its slowest block does, so causal speedups come from many small blocks, not from skipping alone.
- **The biggest win (stage 5, 4×) was mostly a side effect.** Compile-time loop bounds moved `S` out of local memory, which was the goal. Full unrolling also brought the Q row back into registers. Shared-memory stalls fell from 14.1 to 1.2 cycles per instruction, against 0.9 cycles saved on local memory.
- **Warp partitioning doubled occupancy (15% → 29%) and made the 4-head case slower.** The two threads of a row read K/V half-rows 128 B apart, which is the same bank, so loads that used to be free broadcasts became 2-way conflicts. Conflicts went from 17% to 53%, and the MIO-throttle stall from 0.4 to 4.1 cycles. Higher occupancy doesn't help if the extra warps can't issue.
- **fp16 helped through the bank layout, not bandwidth.** Halving the row width put the two threads' reads 16 banks apart. Conflicts fell from 53% to 16%, and issue-slot utilisation rose from 33% to 72%.
- **Double buffering hid global-load latency almost completely.** The long-scoreboard stall dropped 96% (1.27 → 0.06 cycles/inst), and issue-slot utilisation reached 81%.
- **Tensor cores halved the FMAs and bought 3%.** `QKᵀ` uses 3.6% of tensor-pipe capacity. Scalar `P·V` plus the round trip of S through shared memory (61% bank conflicts) now dominate. That points straight at `mma.sync` with register-resident S.
- **Profiling found a race in the first WMMA version.** The end-of-loop `__syncthreads()` from the double-buffered kernel had been dropped, so the next tile's `cp.async` overwrote K/V that slower warps were still reading. Output was wrong whenever a block had more than one warp. `compute-sanitizer --tool racecheck` confirmed it, and the fix is committed.

### Correctness

234 checks across all stages and tile sizes, against fp64 CPU attention references. fp16 stages are compared with a reference computed from the **same fp16-rounded inputs** (tolerance 1e-4), which separates kernel bugs from storage-rounding cost. The worst kernel error is 8.4e-7 on the perf shapes and 2.6e-5 on the adversarial ragged batch. fp16 input rounding alone costs about 1e-3 against an fp32 reference (2.4e-2 on the adversarial case with ×25 key rows).

## Features

### FlashAttention 2: CUDA kernel (`kernels/flash_attention.cuh`)

`flash_attention<H, B_r, B_c, D_K, D_V, causal, T>`. Each line is a commit-level stage in the table above:

- [x] **Tiled forward:** one block owns `B_r` query rows; the `Q` tile is loaded once into shared memory and held for the whole K/V loop.
- [x] **Online softmax:** running `(m, ℓ, O_acc)` per query row, rescaled by `exp(m_old − m_new)` on every tile. `P` is never materialised.
- [x] **Ragged dimensions:** `N` need not divide `B_r` or `B_c`. Short tiles are zero-padded and masked to `−∞`.
- [x] **Causal masking:** per-element mask on diagonal tiles, plus whole-tile skipping when the block's highest query index is below the tile's lowest key index (block-uniform, so no divergence). Fully-masked rows never produce `NaN`.
- [x] **Batching and multi-head attention:** grid `y`/`z` map to head and batch over a `[B, N, H, D]` layout.
- [x] **`S` in registers:** compile-time loop bounds and full unrolling, so there's no local-memory traffic.
- [x] **Warp partitioning:** two threads per query row. Partial `QKᵀ` dot products are combined with `__shfl_xor_sync`, and each thread owns half of `O`.
- [x] **fp16 storage:** `Q`/`K`/`V` stored as `__half` with fp32 accumulation; `exp2f` with `log₂e` folded into the scale.
- [x] **`cp.async` double buffering:** two K/V shared-memory buffers, with tile `i+1` prefetched via `__pipeline_memcpy_async` while tile `i` computes.

### Tensor cores (`kernels/flash_attention_wmma.cuh`)

`S = QKᵀ` on tensor cores with WMMA 16×16×16 fragments. Each warp owns 16 query rows, and `K` fragments are loaded column-major to read the transpose. Supports B_c ∈ {16, 32, 64}.

**Known limits** (from profiling): `S` makes a round trip through shared memory because WMMA fragments are opaque, and `P·V` is still scalar. The next step is `mma.sync` PTX, which has a documented fragment layout. That lets the softmax run directly on the accumulators and feeds `P` from registers into a second MMA for `P·V`.

### Known issue

The double-buffered loop prefetches tile `i+1` before checking whether it will be skipped, so every block loads every K/V tile. Hardware counters show exactly 2× the necessary `cp.async` bytes. It's cheap today (DRAM under 3% utilised) but it's pure waste. The fix is to stop the loop at the last non-skipped tile.

### Reference & verification layer (PyTorch)

| Component | File | Notes |
|---|---|---|
| Attention reference | `reference/attention.py` | `S = QKᵀ·scale`, `P = softmax(S)`, `O = PV`, optional causal mask |
| Tiled FA2 reference | `reference/flash_attention.py` | The exact `(m, ℓ, O_acc)` recurrence the CUDA kernel implements, with an optional per-iteration trace |
| Test cases | `reference/cases.py` | Hand-inspectable 8×4×6, adversarial max-position cases, non-dividing N = 517 with d_k ≠ d_v, N = 4096, batched multi-head, causal and non-causal; fp16-storage variants with true-fp32 references |
| Verification | `reference/check.py` | Tiled reference vs. manual attention **and** vs. `F.scaled_dot_product_attention` |
| Trace + fixture dump | `reference/dump_trace.py` | `.npz` traces for line-by-line kernel debugging; raw `.f32` binaries the CUDA harness loads |
| Multi-head attention | `reference/attention.ipynb` | Batched MHA module, verified against batched SDPA and by a masked-prefix independence check |

The adversarial cases exist because online softmax is only interesting when the running max actually changes. A case where the max lands in the first tile never exercises the rescale path.

## Goals

**Near-term:** finish the optimisation ladder.

- [x] fp16 tile storage with fp32 accumulation
- [x] `cp.async` double buffering of the K/V tiles
- [x] Nsight Compute profiling of every stage
- [ ] Skip-aware prefetch (stop loading tiles that causal masking skips)
- [ ] Tensor cores via `mma.sync` PTX with register-resident `S` (WMMA `QKᵀ` prototype done)
- [ ] `P·V` on tensor cores
- [ ] Swizzled shared-memory layouts to remove the remaining bank conflicts

## Stack

| Layer | Technology |
|---|---|
| Kernel | CUDA C++ (templated, header-only) |
| Reference & verification | PyTorch, NumPy |
| Profiling | Nsight Compute, `compute-sanitizer` |

**Target hardware:** RTX 3060 Ti (Ampere, CC 8.6, 8 GB VRAM, 38 SMs). Development on Windows. The 8 GB budget is a deliberate constraint: most public FlashAttention writeups assume A100/H100-class memory and shared-memory budgets.

## Repository Structure

```
.
├── kernels/
│   ├── flash_attention.cuh       # the templated FA2 forward kernel (through double buffering)
│   ├── flash_attention_wmma.cuh  # WMMA tensor-core QKᵀ variant
│   ├── tuner.cuh                 # B_r × B_c timing sweep
│   └── main.cu                   # loads .f32 fixtures, launches, checks
└── reference/
    ├── attention.py          # ground-truth attention
    ├── flash_attention.py    # tiled online-softmax reference (+ trace)
    ├── cases.py              # shared test cases
    ├── check.py              # reference vs. manual vs. SDPA
    ├── dump_trace.py         # writes traces/ and ../matrices/ fixtures
    └── attention.ipynb       # single-head + multi-head exploration
```

Fixtures are generated, not committed: run `python reference/dump_trace.py` to write `../matrices/<case>/{Q,K,V,O}.f32` before building. Then, from `kernels/` (fixture paths are relative): `nvcc -O3 -arch=sm_86 main.cu -o main && ./main`.

## Methodology

1. **Reference first.** No kernel is written until a PyTorch reference exists and is independently verified against `F.scaled_dot_product_attention`.
2. **Verification at increasing scale:** 8×4×6 (inspectable by hand), then 512, then non-dividing 517, then 4096, before any performance work.
3. **Trace-driven debugging.** The PyTorch reference emits per-iteration `(m, ℓ, O_acc, corr, rowsum)`, so a kernel mismatch is localised to a specific tile iteration rather than guessed at.
4. **Profile after correctness, with a fixed methodology.** Clocks locked to base, caches flushed, a fixed metric set, the median of 5 launches, and each stage compared at the *same* tile size and workload as the one before it. Claims are cross-checked against the compiled SASS.

## Status Disclaimer

Active learning project. The kernel is verified for numerical correctness at every stage; timings reflect this specific GPU (RTX 3060 Ti) and shouldn't be assumed to generalise.
