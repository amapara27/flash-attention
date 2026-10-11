#!/usr/bin/env bash
# Writes each stage's kernel to stages/<stage>/kernel.cuh, straight from the commit that finished it.
# Stages 1-3 lived in a single .cu with main(); only the kernel (the first N lines) is taken.
cd "$(dirname "$0")"

extract() {   # extract <stage dir> <commit> <path> [first N lines]
    mkdir -p "stages/$1"
    if [[ -n "$4" ]]; then
        git show "$2:$3" | head -n "$4" > "stages/$1/kernel.cuh"
    else
        git show "$2:$3" > "stages/$1/kernel.cuh"
    fi
    echo "$1  <-  $2:$3${4:+ (first $4 lines)}"
}

extract 01_tiled             c4ad580 kernels/flash-attention.cu  155
extract 02_causal_mask       7de13d3 kernels/flash-attention.cu  169
extract 03_tile_skipping     81bc58e kernels/flash-attention.cu  180
extract 04_multihead         3249aeb kernels/flash-attention.cuh
extract 05_s_registers       b890554 kernels/flash-attention.cuh
extract 06_warp_partitioning bde48e3 kernels/flash-attention.cuh
extract 07_fp16_smem         5687ebd kernels/flash-attention.cuh
extract 08_double_buffering  c2d1ecf kernels/flash-attention.cuh
extract 09_wmma_qk           5959273 kernels/flash_attention_wmma.cuh
extract 09b_wmma_qk_fixed    9f657c5 kernels/flash_attention_wmma.cuh
