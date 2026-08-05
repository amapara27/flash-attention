#pragma once

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include "flash-attention.cuh"

// tile size autotuner
template <int B_r, int B_c>
void tuner(float *dQ, float *dK, float *dV, float *dO, const vector<float> &hRef, int N, int reps = 5) {
    constexpr int smem = (B_r * 64 + B_c * 64 + B_c * 64) * sizeof(float);

    if constexpr (smem > 48 * 1024) {
        printf("%3d,%3d,SKIP,,%d\n", B_r, B_c, smem);
        return;
    }

    int blocks = CEIL_DIV(N, B_r);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    flash_attention<B_r, B_c, 64, 64, true><<<blocks, B_r>>>(dQ, dK, dV, dO, N);
    cudaDeviceSynchronize();

    vector<float> times;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(start);
        flash_attention<B_r, B_c, 64, 64, true><<<blocks, B_r>>>(dQ, dK, dV, dO, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    sort(times.begin(), times.end());
    float median = times[reps / 2];

    // correctness
    vector<float> hO(N * 64);
    cudaMemcpy(hO.data(), dO, N * 64 * sizeof(float), cudaMemcpyDeviceToHost);
    float err = 0.0f;
    for (size_t i = 0; i < hO.size(); ++i)
        err = fmaxf(err, fabsf(hO[i] - hRef[i]));

    printf("%3d,%3d,%.3f,%.2e,%d\n", B_r, B_c, median, err, smem);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}