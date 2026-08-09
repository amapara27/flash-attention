#pragma once

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include "flash-attention.cuh"

// tile size autotuner
template <int H, int B_r, int B_c, int D_K = 64, int D_V = 64>
void tuner(float *dQ, float *dK, float *dV, float *dO, const std::vector<float> &hRef,
           int B, int N, int reps = 20) {
    constexpr int smem = (B_r * D_K + B_c * D_K + B_c * D_V) * sizeof(float);

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
    flash_attention<H, B_r, B_c, D_K, D_V, true><<<grid, B_r * 2>>>(dQ, dK, dV, dO, N);

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
        flash_attention<H, B_r, B_c, D_K, D_V, true><<<grid, B_r>>>(dQ, dK, dV, dO, N);
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
    for (size_t i = 0; i < hO.size(); ++i)
        err = fmaxf(err, fabsf(hO[i] - hRef[i]));

    printf("%3d,%3d,%.3f,%.2e,%d\n", B_r, B_c, median, err, smem);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}