#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "tuner.cuh"
#include "flash-attention.cuh"

// load reference matrices into vectors
std::vector<float> load(const char *path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { printf("FATAL: %s has %zu floats, want %zu\n", path, got, n); exit(1); } // verifies correct element amount
    return v;
}

// checker
void checker(const std::vector<float> &O, const std::vector<float> &O_ref, size_t n, int d_v) {
    float max_abs = 0.0f;
    int bad = -1;

    // indexes first miscalculation
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(O[i] - O_ref[i]);
        if (d > max_abs) max_abs = d;
        if (bad < 0 && d > 1e-5f) bad = (int)i;
    }

    printf("max abs err %.3e\n", max_abs);
    if (bad >= 0)
        printf("FAIL: first at [%d, %d] got %.8f want %.8f\n",
               bad / d_v, bad % d_v, O[bad], O_ref[bad]);
    else
        printf("PASS\n");
}


int main() {
    // matrix dims
    const int N = 4096, b = 1, h = 1, d_k = 64, d_v = 64;

    const size_t qk_elems = (size_t)b * N * h * d_k;
    const size_t vo_elems = (size_t)b * N * h * d_v;

    // load reference matricss
    auto hQ   = load("../matrices/massive_causal/Q.f32", qk_elems);
    auto hK   = load("../matrices/massive_causal/K.f32", qk_elems);
    auto hV   = load("../matrices/massive_causal/V.f32", vo_elems);
    auto hRef = load("../matrices/massive_causal/O.f32", vo_elems);

    // allocate device mem
    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, qk_elems * sizeof(float));
    cudaMalloc(&dK, qk_elems * sizeof(float));
    cudaMalloc(&dV, vo_elems * sizeof(float));
    cudaMalloc(&dO, vo_elems * sizeof(float));

    cudaMemcpy(dQ, hQ.data(), qk_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), qk_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), vo_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dO, 0, vo_elems * sizeof(float));

    // launch kernel with autotuner - B_r = 64. B_c = 32 was the best result
    // printf("B_r,B_c,ms,max_err,smem_bytes\n");
    // tuner<h,  16,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  16,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  16,  64>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  32,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  32,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  32,  64>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  64,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  64,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  64,  64>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h, 128,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h, 128,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h, 128,  64>(dQ, dK, dV, dO, hRef, b, N);

    const int b_r = 64;
    const int b_c = 32;

    // x = query tiles, y = heads, z = batches
    dim3 grid(CEIL_DIV(N, b_r), h, b);

    // launch kernel with template dims
    flash_attention<h, b_r, b_c, d_k, d_v, true><<<grid, b_r>>>(dQ, dK, dV, dO, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceSynchronize();

    // allocate + copy output matrix to host
    std::vector<float> hO(vo_elems);
    cudaMemcpy(hO.data(), dO, vo_elems * sizeof(float), cudaMemcpyDeviceToHost);

    // run checker
    checker(hO, hRef, vo_elems, d_v);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    return 0;
}