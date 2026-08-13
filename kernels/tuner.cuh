#pragma once

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "flash-attention.cuh"
#include "flash-attention-wmma.cuh"

// tile size autotuner
// defaults to float unless half passed in
template <int H, int B_r, int B_c, int D_K = 64, int D_V = 64, bool causal, typename T = float>
void tuner(T *dQ, T *dK, T *dV, float *dO, const std::vector<float> &hRef, int B, int N, float tol = 3e-3f, int reps = 20) {
    constexpr int smem = (B_r * D_K + 2 * B_c * D_K + 2 * B_c * D_V) * sizeof(__half) + B_r * B_c * sizeof(float);

    if constexpr (smem > 48 * 1024) {
        printf("%3d,%3d,SKIP,,%d\n", B_r, B_c, smem);
        return;
    }

    // x = query tiles, y = heads, z = batches
    dim3 grid(CEIL_DIV(N, B_r), H, B);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    flash_attention_wmma<H, B_r, B_c, D_K, D_V, causal><<<grid, B_r * 2>>>(dQ, dK, dV, dO, N);

    // a launch that fails (smem over limit, bad config) is silent
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        printf("%3d,%3d,LAUNCHFAIL,%s,%d\n", B_r, B_c, cudaGetErrorString(launch), smem);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }
    cudaDeviceSynchronize();

    std::vector<float> times;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(start);
        flash_attention_wmma<H, B_r, B_c, D_K, D_V, causal><<<grid, 2 * B_r>>>(dQ, dK, dV, dO, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    float median = times[reps / 2];

    // correctness -- full [B, N, H, D_V] output, not one head
    std::vector<float> hO((size_t)B * N * H * D_V);
    cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float err = 0.0f;

    for (size_t i = 0; i < hO.size(); ++i){
        err = fmaxf(err, fabsf(hO[i] - hRef[i]));
    }

    
    if (err > tol) {
        printf("%3d,%3d,FAIL,%.2e,%d\n", B_r, B_c, err, smem);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    printf("%3d,%3d,%.3f,%.2e,%d\n", B_r, B_c, median, err, smem);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}