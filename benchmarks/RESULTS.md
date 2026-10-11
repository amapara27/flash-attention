# FA2 forward on an RTX 3060 Ti: results

Data: `results/summary.csv` (one row per stage × workload × tile, median of each metric over 5 profiled launches),
`results/correctness.csv` (every gate check), `results/timing.csv` (boost-clock wall time). The raw Nsight Compute reports
(~4.7 GB) aren't committed; `run_ncu.py` regenerates them (see `README.md`).

## Method
- **Correctness first:** 234 checks against fp64 CPU references; fp16 stages are compared with a reference computed from the *same*
  fp16-rounded inputs (tol 1e-4). Re-run after profiling with identical results. The only failures are committed stage 9 (race).
- **ncu time:** clocks locked to base, caches flushed, median of 5 launches. This is the primary number because it's deterministic.
- **wall time:** cudaEvent at boost clocks, median of 50 (typically 1.3× faster than ncu).
- **TFLOPS:** useful FLOPs, `4·N²·d·H`, halved for causal. For reference, the RTX 3060 Ti's rated FP32 peak is about 16 TFLOPS.
- Workloads: **A0** = 1 head, N=4096, d=64, non-causal · **A** = same, causal · **B** = 4 heads, causal.

## Progression (best tile per stage, chosen by ncu time)

| # | Stage | A: tile · ncu ms · wall ms | B: tile · ncu ms · wall ms · TFLOPS (wall) |
|---|---|---|---|
| 1 | tiled (non-causal, A0) | 128×32 · 3.42 · 2.68 | — |
| 2 | + causal mask | 128×32 · 8.82 · 6.53 | — |
| 3 | + tile skipping | 64×32 · 7.39 · 5.63 | — |
| 4 | + multi-head | 64×32 · 7.64 · 5.80 | 128×32 · 18.20 · 13.14 · 0.65 |
| 5 | S in registers | 128×32 · 3.05 · 2.46 | 128×32 · 4.55 · 3.40 · 2.53 |
| 6 | warp partitioning | 64×16 · 2.70 · 2.11 | 128×32 · 5.26 · 3.99 · 2.19 |
| 7 | fp16 storage | 128×32 · 2.16 · 1.37 | 64×16 · 3.80 · 2.86 · 3.00 |
| 8 | cp.async double buffer | 32×16 · 1.02 · 0.77 | 64×16 · 2.74 · 2.12 · 4.04 |
| 9b | WMMA QKᵀ (fixed) | 32×16 · 0.82 · 0.58 | 32×16 · 2.66 · 1.88 · 4.56 |

**End to end (causal):** A 8.82 → 0.82 ms (**10.7×**) · B 18.20 → 2.66 ms (**6.8×**), 4.6 TFLOPS at boost clocks.

## Key findings
1. **It's never DRAM-bound.** Max DRAM utilisation across all 165 profiled configs is 2.8%. FlashAttention removes the HBM
   bottleneck by design, so every bottleneck in this study is on-chip: shared-memory banks, the issue rate, and latency.
2. **Turning on the causal mask costs 2.6×, and the mask isn't the reason (stage 2).** The compiler changed register allocation and stopped keeping
   the Q row in registers. It re-loads Q from shared memory for every key column, with 8-way bank conflicts.
3. **Halving the work saved 0% (stage 3)** at 128×32: 32 blocks on 38 SMs, and the last block needs every tile anyway. The
   kernel finishes when its slowest block does.
4. **The biggest win was mostly accidental (stage 5, 4×).** Moving S out of local memory was the goal. Full unrolling also restored
   Q-in-registers, which removed most of the shared-memory stalls (short-scoreboard 14.1 → 1.2 cycles/inst).
5. **Doubling occupancy made it slower (stage 6).** Splitting rows across two threads put their K/V reads 128 B apart, which is the same
   bank, and turned broadcast loads into 2-way conflicts.
6. **fp16 helped through bank layout, not bandwidth (stage 7).** Half-rows of 64 B land in different banks. Conflicts fall from 53% to 16%, and
   issue-slot use rises from 33% to 72%.
7. **Double buffering works as designed (stage 8).** Long-scoreboard stalls fall 96% and issue-slot use reaches 81%. Counters also confirm it
   loads every K/V tile, including skipped ones (exactly 2× the bytes).
8. **Tensor cores bought 3% (stage 9b).** QKᵀ uses 3.6% of the tensor pipe. Scalar P·V plus the S round trip through
   shared memory (61% conflicts) dominate. The committed stage-9 kernel had a race and produced wrong output whenever
   a block had more than one warp.
9. **The best tile shrinks as the kernel improves:** 128×32 in early stages, 32×16 or 64×16 at the end. Once per-tile work is cheap,
   more blocks (better causal load balance) matter more. B_c = 64 is never competitive after stage 5 because of register pressure.

---

## Stages

Each section lists what the stage added, what the profile showed (best tile unless a same-tile comparison is stated), and
what was still missing. Per-tile times for every stage are in [Tile sweeps](#tile-sweeps).

### 1 · Tiled: `c4ad580`
`flash_attention<B_r, B_c, D_K, D_V>` · grid `⌈N/B_r⌉` · block `B_r` (1 thread per row) · smem `256·(B_r + 2·B_c)` B · fp32 · 1 head, non-causal

**Adds:** the first complete FA2 forward. Each block owns `B_r` query rows, loads its Q tile into shared memory once, and streams every
K/V tile through. Online softmax keeps a running max `m`, sum `ℓ`, and unnormalised `O_acc` per row, rescaled by `exp(m_old − m_new)`
when a tile raises the max. `P` is never stored, and `O = O_acc / ℓ` is written once at the end.

**Profile (128×32, 3.42 ms):** 156 registers, 8.3% achieved occupancy, 0.42 waves (32 blocks on 38 SMs), 18% issue-slot use, and 134 MB
of local-memory loads (`S[]`). DRAM reads 3.2 MB, which is Q+K+V once; everything else hits L2 (97.8%).
- The GPU is mostly idle: 4,096 threads on a chip that holds about 58k.
- B_r *is* the thread count, so B_r = 16 means half-empty warps. At 16×64 shared memory fits only two blocks per SM: 4.2% occupancy, 53 ms.

**Lacking:** causal masking; `S[]` in local memory (runtime loop bound); 32-way-pattern bank conflicts on Q reads (every lane
reads its own row); one head only; fp32 storage; synchronous loads.

### 2 · Causal mask: `7de13d3`
Adds a `causal` template flag: scores with key index > query index become `−∞`. A `masked_out` guard avoids `exp(−∞ − (−∞)) = NaN`.

**Result:** with `causal=false` it matches stage 1 (3.44 ms), so the unused mask is free. With `causal=true` it takes **8.82 ms, 2.6× slower**
for the *same* 2.22 G FFMAs. There's no skipping yet, and `masked_out` never fires because every row sees key 0 in tile 0.

| 128×32 | non-causal | causal |
|---|---|---|
| Registers | 156 | **94** |
| Shared loads per warp per K/V tile | 1,040 | **1,536** |
| Bank-conflict rate | 17% | **78%** |
| Short-scoreboard stall (cycles/inst) | 0.46 | **8.3** |

1,040 = 16 (Q row, loaded once and kept in registers) + 512 (K) + 512 (V). The causal kernel issues exactly 512 more: the **Q row
re-loaded for every one of the 32 key columns**, because at 94 registers nvcc stops keeping it resident. Those loads are 8-way
conflicted (rows 256 B apart, so every lane hits the same banks). Static SASS is the same size; register allocation is what changed.

**Lacking:** every masked tile is still loaded and computed (about 2× the necessary work).

### 3 · Tile skipping: `81bc58e`
Skips a K/V tile when its lowest key index exceeds the block's highest query index. The condition depends only on the block, so
there's no divergence. This commit also added the first tile tuner (its "best 64×32" reproduces).

**Result:** 7.39 ms at 64×32 (1.19× vs stage 2). FFMAs and instructions halve, but **at 128×32 the time doesn't move (8.82 → 8.78 ms)**.
There are 32 blocks for 38 SMs, so each block has its own SM, and the last block (rows 3968–4095) needs every tile regardless. The kernel finishes when its
slowest block does. At 64×32 (64 blocks) some SMs run two blocks, so shorter blocks finally help (−20%).

**Lacking:** load imbalance across the causal triangle; the stage-2 Q-reload problem (98 registers, 77.5% conflicts).

### 4 · Multi-head: `3249aeb`
`flash_attention<H, …>` · grid `(⌈N/B_r⌉, H, B)` over a `[B, N, H, D]` layout with per-block pointer offsets.

**Result:** about 3% overhead on 1 head (7.64 vs 7.39 ms), with the same registers. 4 heads: **18.20 ms at 128×32**, which is 4× the work in 2.0× the time.
The grid grows from 32 to 128 blocks and waves from 0.42 to 1.68. This is the 4-head baseline: warp latency is **20.4 cycles/inst, 14.1 of them
short-scoreboard**, with 77.6% conflicts, 9% issue-slot use, and 277 MB of local loads.

**Lacking:** the same Q-reload and local-memory problems, now at scale.

### 5 · S in registers: `b890554`
Loop bounds became compile-time (`col < B_c`) with `#pragma unroll`. Ragged tiles are handled by zero-filling K/V and masking columns past N to `−∞`.

**Result: 4.0× on 4 heads (18.20 → 4.55 ms), 2.5× on 1 head. The biggest jump in the progression.**

| 4 heads, 128×32 | stage 4 | stage 5 |
|---|---|---|
| Local-memory loads | 277 MB | **0** |
| SASS `LDL`/`STL` · static `FFMA` | 9/1 · 468 | 0/0 · 4,561 (fully unrolled) |
| Shared loads per warp-tile | 1,536 | **1,040** |
| Bank-conflict rate | 77.6% | **17.3%** |
| Warp latency: short-scoreboard / long-scoreboard | 14.1 / 2.4 | **1.2 / 1.4** |
| Issue-slot use · registers | 9% · 98 | 35% · 151 |

There were two fixes in one commit. The intended one moved `S` out of local memory. The accidental one: full unrolling let nvcc keep the Q row in
registers again, which undid stage 2. The stalls show which mattered: shared-memory stalls fell 12.9 cycles per instruction, local-memory
stalls 0.9.

**Lacking:** registers grow with B_c. B_c = 64 needs 254 (the hardware maximum is 255) and gets 4–8% occupancy. Still one thread per row.

### 6 · Warp partitioning: `bde48e3`
Two adjacent lanes share each row (block = 2·B_r). For `QKᵀ` each lane computes half of every dot product, combined with
`__shfl_xor_sync(…, 1)`. For `P·V` each lane owns half the output columns (`O_acc[32]`).

**Result:** 1.13× faster on 1 head, but **16% slower on 4 heads** at the best tile.

| 4 heads, 128×32 | stage 5 | stage 6 |
|---|---|---|
| Achieved occupancy | 15.3% | **28.6%** (commit message: 28.77%) |
| Issue-slot use | 34.6% | 33.1% |
| Bank-conflict rate | 17.3% | **53.0%** |
| MIO-throttle stall | 0.37 | **4.08** |
| Time | 4.55 ms | **5.26 ms** |

The two lanes' K/V half-rows are 32 floats (128 B) apart. 128 B is one full bank cycle, so both map to the same bank, and the free broadcast loads
became 2-way conflicts. It also adds 32 shuffles per tile, and the `exp` now runs twice per row (XU pipe +65%). The extra warps
stall on the same shared-memory pipe. The stage does help where stage 5 had half-empty warps: 16×16 goes from 26.6 to 13.3 ms.

**Lacking:** the new conflicts; softmax computed twice; fp32 storage; synchronous loads.

### 7 · fp16 storage: `5687ebd`
Q/K/V are `__half` in global and shared memory, with fp32 math and accumulators. The commit also folded `log₂e` into the scale and switched `expf` to `exp2f`.

**Result:** 1.38× on 4 heads (3.80 ms at 64×16), 1.25× on 1 head. At the same tile (4 heads, 128×32: 5.26 → 4.03 ms) conflicts go from 53% to **16%**,
MIO throttle from 4.08 to 0.04, and issue-slot use from 33% to **72%**. L2 and DRAM bytes halve, but that's irrelevant, since DRAM never exceeds 3%. The
gain comes from the row width: fp16 half-rows are 64 B, so the two lanes' reads land 16 banks apart. At B_c = 16 it needs 80 registers (50%
theoretical occupancy), which is why the best tile moves to 64×16. Precision: 1.0e-3 against an fp32 reference, all of it input rounding.

**Lacking:** confounded with `exp2f`, and the profile can't separate them. Instructions *rise* (258 M → 389 M, half→float converts), the math
is still scalar, and loads are still synchronous.

### 8 · cp.async double buffering: `c2d1ecf`
Two K/V shared-memory buffers. A prologue loads tile 0; then each iteration issues `__pipeline_memcpy_async` (16 B/lane, straight to shared
memory) for tile `i+1`, waits for tile `i` with `wait_prior(1)`, and computes. It runs with `T = __half`.

**Result:** 1.39× on 4 heads (2.74 ms at 64×16), **2.12× on 1 head** (1.02 ms). At the same tile (4 heads, 64×16), long-scoreboard stalls go from 1.27 to
**0.055 cycles/inst (−96%)**, issue-slot use from 67% to **81%** (the highest in the study), and the FMA pipe from 49% to 63%. Global-load efficiency goes from 2 to about 15.6
sectors/request (ideal 16). The single head gains most because it had the fewest warps to hide latency with. Correctness: 72/72 including
ragged shapes, and racecheck is clean. The "failing references" in the commit message were fp16-vs-fp32 rounding, fixed by `8af1448`.

**Bug, confirmed by hardware counters:** tile `i+1` is prefetched *before* the skip check, so every block loads every tile. On 4 heads that's exactly
268,435,456 B of `cp.async` (256 blocks × 256 tiles × 4 KB), and DRAM and L2 traffic double compared with stage 7. It's cheap today, but pure waste.

**Lacking:** prefetch-before-skip; Q still loaded synchronously; scalar math; doubled K/V shared memory.

### 9 · WMMA tensor cores for QKᵀ: `5959273`, fixed in `9f657c5`
`QKᵀ` on tensor cores with `nvcuda::wmma` 16×16×16 fragments. Each warp owns 16 query rows, and K is loaded col-major (that gives Kᵀ for free). S is stored to
`S_smem` (fragments are opaque), and then the scalar mask, softmax and `P·V` run as before.

**The committed kernel is wrong whenever a block has more than one warp** (error up to 0.29). racecheck shows about 10k hazards: the end-of-loop
`__syncthreads()` from stage 8 was dropped, so the next tile's `cp.async` overwrites K/V that slower warps are still reading. The fix (`9f657c5`)
restores the barrier, zeroes all `B_c/16` accumulators instead of hardcoding two (which unlocks B_c = 16 and 64), and adds `__syncwarp()`
after `store_matrix_sync`. The fixed kernel passes 20/20 configs, 10 out of 10 repeats each, and racecheck is clean.

**Result (fixed):** 2.66 ms on 4 heads (**1.03×** vs stage 8), 0.82 ms on 1 head (1.24×).

| 4 heads, 32×16 | stage 8 | 9b |
|---|---|---|
| FFMA executed | 4.34 G | **2.17 G** |
| Tensor-pipe use | 0% | **3.6%** |
| Issue-slot use | 80% | 51% |
| Bank-conflict rate | 25% | **61%** |
| Time | 3.11 ms | 2.66 ms |

Amdahl's law: `P·V` (half the FLOPs) is still scalar. And S takes a round trip through shared memory: store, scale/mask, row max, `P·V`, with
row-wise reads that conflict. The barrier itself costs about 3% (16×32, where the race can't occur: 4.25 → 4.38 ms).

**Lacking:** `mma.sync` with register-resident S (softmax on the accumulators, P fed from registers into a second MMA for `P·V`).

---

## Tile sweeps

ncu ms per B_r × B_c. FAIL = wrong output (not profiled); — = not run (over 48 KB shared memory or not applicable).

**A0** (1 head, non-causal)

| Stage | 16×16 | 16×32 | 16×64 | 32×16 | 32×32 | 32×64 | 64×16 | 64×32 | 64×64 | 128×16 | 128×32 | 128×64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 14.37 | 27.96 | 53.45 | 8.22 | 8.06 | 15.93 | 5.01 | 4.88 | 4.81 | 3.64 | **3.42** | — |
| 2 | 14.41 | 28.03 | 53.17 | 8.25 | 8.57 | 15.98 | 5.03 | 4.89 | 4.84 | 3.66 | **3.44** | — |

**A** (1 head, causal)

| Stage | 16×16 | 16×32 | 16×64 | 32×16 | 32×32 | 32×64 | 64×16 | 64×32 | 64×64 | 128×16 | 128×32 | 128×64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 18.02 | 31.67 | 58.93 | 11.99 | 11.98 | 21.43 | 9.11 | 9.27 | 9.81 | 8.85 | **8.82** | — |
| 3 | 15.91 | 16.62 | 24.13 | 10.62 | 10.58 | 12.14 | 7.44 | **7.39** | 7.47 | 8.84 | 8.78 | — |
| 4 | 16.93 | 18.02 | 26.23 | 11.11 | 11.07 | 12.83 | 7.68 | **7.64** | 7.74 | 8.96 | 8.95 | — |
| 5 | 14.13 | 15.26 | 23.66 | 7.73 | 7.90 | 9.74 | 4.48 | 4.53 | 5.64 | 3.07 | **3.05** | — |
| 6 | 7.13 | 7.91 | 11.43 | 4.04 | 4.19 | 4.69 | **2.70** | 2.75 | 2.73 | 2.91 | 2.81 | — |
| 7 | 7.03 | 7.35 | 8.14 | 4.05 | 4.17 | 5.37 | 2.61 | 2.54 | 3.76 | 2.30 | **2.15** | 3.07 |
| 8 | 1.18 | 1.85 | 5.50 | **1.02** | 1.23 | 3.30 | 1.05 | 1.26 | 2.27 | 1.37 | 1.33 | 2.37 |
| 9 | — | 1.34 | — | — | **0.94** | — | — | FAIL | — | — | FAIL | — |
| 9b | 0.93 | 1.36 | 2.39 | **0.82** | 0.95 | 1.71 | 0.94 | 1.07 | — | 1.28 | 1.45 | — |

**B** (4 heads, causal)

| Stage | 16×16 | 16×32 | 16×64 | 32×16 | 32×32 | 32×64 | 64×16 | 64×32 | 64×64 | 128×16 | 128×32 | 128×64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 34.53 | 48.35 | 83.60 | 23.11 | 23.86 | 35.52 | 19.00 | 18.55 | 21.04 | 21.19 | **18.20** | — |
| 5 | 26.56 | 39.49 | 75.69 | 12.51 | 13.96 | 25.78 | 5.94 | 6.92 | 10.32 | 4.62 | **4.55** | — |
| 6 | 13.30 | 20.38 | 36.54 | 6.95 | 7.76 | 12.83 | 5.76 | 5.65 | 5.77 | 5.60 | **5.26** | — |
| 7 | 9.43 | 12.51 | 20.44 | 4.75 | 6.14 | 9.65 | **3.80** | 4.40 | 5.64 | 4.08 | 4.03 | 6.45 |
| 8 | 3.43 | 4.77 | 17.95 | 3.11 | 3.28 | 9.34 | **2.73** | 2.97 | 5.37 | 3.52 | 2.93 | 5.53 |
| 9 | — | **4.25** | — | — | FAIL | — | — | FAIL | — | — | FAIL | — |
| 9b | 3.06 | 4.38 | 7.95 | **2.66** | 3.12 | 4.67 | 2.73 | 3.19 | — | 3.18 | 3.80 | — |

---

## What the data says to do next
1. Fix the prefetch-before-skip bug (stage 8) by limiting the K/V loop to the last non-skipped tile.
2. Use `mma.sync` with register-resident S: no `S_smem` round trip and no conflicts; softmax on the accumulators.
3. Put P·V on tensor cores too: the other half of the FLOPs.
4. Swizzle or pad the shared-memory layouts to address the conflicts that remain.
