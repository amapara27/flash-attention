# FlashAttention 2 — Forward Kernel from Scratch

![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-reference%20%2B%20verification-EE4C2C?logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RTX%203060%20Ti%20(Ampere%2C%20CC%208.6)-blue)

A from-scratch CUDA implementation of the FlashAttention 2 forward pass, built for consumer Ampere. No cuBLAS, no cuDNN, no PyTorch in the kernel layer — the kernel is hand-written and verified element-wise against PyTorch references.

The repo is the full derivation chain, not just the final kernel: a plain attention reference → a tiled online-softmax reference in PyTorch (with a per-iteration trace) → the fused CUDA kernel that implements the same recurrence in registers.

## Current Status

**FA2 forward is implemented, verified, and being optimized rung by rung.** The kernel runs batched multi-head attention with causal masking at N = 4096, d_k = d_v = 64, with fp16 tile storage, `cp.async` double buffering, and warp partitioning. Next up: tensor cores via `mma.sync` PTX.

Everything below under **Features** is real today. **Goals** describes where this is headed.

## Features

### FlashAttention 2 — CUDA kernel (`kernels/flash-attention.cuh`)

Single templated kernel, `flash_attention<B_r, B_c, D_K, D_V, causal>`:

- [x] Single-tile forward (N ≤ B_c, no inner loop)
- [x] Multi-tile forward with online-softmax rescaling — running `(m, ℓ, O_acc)` per query row, corrected by `exp(m_old − m_new)` on every tile
- [x] Multi-block — one block owns `B_r` query rows, one thread owns one row; `Q` tile loaded once into shared memory and held for the whole K/V loop
- [x] `K`/`V` tiles staged in shared memory with coalesced loads; `S` and `O_acc` live in registers, `P` is never materialized
- [x] Ragged dimensions — `N` need not divide `B_r` or `B_c`; out-of-range threads and short tiles are guarded
- [x] Causal masking — per-element mask on diagonal tiles plus whole-tile skipping when the block's highest query index is below the tile's lowest key index (block-uniform, so no warp divergence)
- [x] Fully-masked rows handled without producing `NaN` from `-inf − (−inf)`
- [x] Batching and multi-head attention — grid `y`/`z` map to batch and head, pointer offsets locate each `(batch, head)` slice
- [x] Numerical verification against the PyTorch reference at N = 4096 (`kernels/main.cu`)
- [x] Tile-size tuning — timing sweep over `B_r × B_c` ∈ {16, 32, 64, 128} × {16, 32, 64} (`kernels/tuner.cuh`), median of 5 runs with `cudaEvent` timing, shared-memory budget checked at compile time and oversized configs skipped
- [x] `S` kept in registers with compile-time indexing (no local-memory spill)
- [x] Warp partitioning — two threads per query row, each owning half the columns; raised achieved occupancy ~13% to 28.8% (33% theoretical)
- [x] fp16 tile storage — `Q`/`K`/`V` shared-memory tiles stored as `__half`, accumulation in fp32; fp32 references of the fp16 inputs quantify the precision cost
- [x] `cp.async` double buffering — `K`/`V` shared-memory buffers are doubled; the first tile loads before the K/V loop, then each iteration prefetches the next tile into the `(i + 1) % 2` buffer via CUDA's asynchronous pipeline (`__pipeline_memcpy_async`) while the current tile computes, so warps no longer stall waiting on tile loads

Current config on the target GPU: **`B_r = 128`, `B_c = 32`** (two threads per row → 256-thread blocks).

**Profiling after double buffering:** ~2.9 ms duration, SM throughput nearly doubled from 30.81% to 59.26%, and warp latency dropped from 11.10 to 4.34 cycles.

### Tensor cores — work in progress

A first pass in `kernels/flash_attention_wmma.cuh` implemented `S = QKᵀ` with WMMA 16×16×16 fragments (one warp owns 16 query rows, `K` fragments loaded column-major to read the transpose). Rather than finishing the WMMA path, the plan is to move straight to `mma.sync` in PTX for finer control — currently learning MMA to integrate tensor cores into the main kernel.

Not yet done: full Nsight Compute roofline analysis (achieved bandwidth vs. roofline).

### Reference & verification layer (PyTorch)

| Component | File | Notes |
|---|---|---|
| Attention reference | `reference/attention.py` | `S = QKᵀ·scale`, `P = softmax(S)`, `O = PV`, optional causal mask |
| Tiled FA2 reference | `reference/flash_attention.py` | The exact `(m, ℓ, O_acc)` recurrence the CUDA kernel implements, with an optional per-iteration trace |
| Test cases | `reference/cases.py` | Hand-inspectable 8×4×6, adversarial max-position cases (max in first/middle/last tile), N = 512, non-dividing N = 517, N = 4096, batched multi-head cases, causal and non-causal; fp16-storage variants with true-fp32 references to isolate the precision cost of `__half` tiles |
| Verification | `reference/check.py` | Tiled reference vs. manual attention **and** vs. `F.scaled_dot_product_attention` |
| Trace + fixture dump | `reference/dump_trace.py` | `.npz` traces for line-by-line kernel debugging; raw `.f32` binaries the CUDA harness loads |
| Multi-head attention | `reference/attention.ipynb` | Batched MHA module, verified against batched SDPA and by a masked-prefix independence check |

The adversarial cases exist because online softmax is only interesting when the running max actually changes — a case where the max lands in the first tile never exercises the rescale path.

## Goals

**Near-term:** finish the optimization ladder over the working kernel —

- [x] fp16 tile storage with fp32 accumulation
- [x] `cp.async` double-buffering of the K/V tiles
- [ ] Tensor cores via `mma.sync` PTX (WMMA prototype for `QKᵀ` exists; going straight to MMA)
- [ ] Nsight Compute roofline: achieved bandwidth vs. roofline (occupancy already measured)

**Long-term — the Blaze inference engine (paused, will resume):**

- Remaining transformer primitives in CUDA — RMSNorm, RoPE, SwiGLU FFN, sampling
- A standalone Rust inference runtime: FFI bindings, safetensors loading, KV cache, decode loop
- FlashDecoding — split-K over the KV cache, since one query row can't fill 38 SMs during single-token decode
- **Flagship milestone:** coherent generation from a real open-weights checkpoint (~1B params, fits in 8 GB) with no Python in the serving path
- INT8/INT4 quantization with measured perplexity impact — forced by the 8 GB budget
- Backward passes, enabling training on this project's own kernels

## Stack

| Layer | Technology |
|---|---|
| Kernel | CUDA C++ (templated, header-only) |
| Reference & verification | PyTorch, NumPy |
| Future runtime (Blaze, paused) | Rust (FFI over the CUDA kernels) |

**Target hardware:** RTX 3060 Ti (Ampere, CC 8.6, 8 GB VRAM, 38 SMs). Development on Windows. The 8 GB budget is a deliberate constraint — most public FlashAttention writeups assume A100/H100-class memory and shared-memory budgets.

## Repository Structure

```
.
├── kernels/
│   ├── flash_attention.cuh       # the templated FA2 forward kernel
│   ├── flash_attention_wmma.cuh  # tensor-core variant (WIP)
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

Fixtures are generated, not committed — run `python reference/dump_trace.py` to write `../matrices/<case>/{Q,K,V,O}.f32` before building.

## Methodology

1. **Reference first.** No kernel is written until a PyTorch reference exists and is independently verified against `F.scaled_dot_product_attention`.
2. **Verification at increasing scale** — 8×4×6 (inspectable by hand), then 512, then non-dividing 517, then 4096 — before any performance work.
3. **Trace-driven debugging.** The PyTorch reference emits per-iteration `(m, ℓ, O_acc, corr, rowsum)`, so a kernel mismatch is localized to a specific tile iteration rather than guessed at.
4. **Profile after correctness.**

## Status Disclaimer

Active learning project. The kernel is verified for numerical correctness at each step; timings reflect this specific GPU (RTX 3060 Ti) and shouldn't be assumed to generalize.
