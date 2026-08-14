#include <cstdio>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "tuner.cuh"
#include "flash_attention.cuh"

// load reference matrices into vectors
std::vector<float> load(const char *path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { printf("FATAL: %s has %zu floats, want %zu\n", path, got, n); exit(1); } // verifies correct element amount
    return v;
}

// converts fp32 to fp16 on load
std::vector<__half> load_fp16(const char *path, size_t n) {
    std::vector<float> f(n);
    FILE* file = fopen(path, "rb");
    if (!file) { printf("FATAL: cannot open %s\n", path); exit(1); }
    size_t got = fread(f.data(), sizeof(float), n, file);
    fclose(file);
    if (got != n) { printf("FATAL: %s has %zu floats, want %zu\n", path, got, n); exit(1); }

    std::vector<__half> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = __float2half(f[i]);
    return h;
}

// checker
void checker(const std::vector<float> &O, const std::vector<float> &O_ref, size_t n, int d_v) {
    float max_abs = 0.0f;
    int bad = -1;

    // indexes first miscalculation
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(O[i] - O_ref[i]);
        if (d > max_abs) max_abs = d; 
        // 3e-3 due to fp16 casting, was 1e-5 previously
        if (bad < 0 && d > 3e-3f) bad = (int)i;
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
    const int N = 4096, b = 1, h = 4, d_k = 64, d_v = 64;
    // const int N = 517, b = 2, h = 3, d_k = 64, d_v = 48;

    const size_t qk_elems = (size_t)b * N * h * d_k;
    const size_t vo_elems = (size_t)b * N * h * d_v;

    const bool causal_masking = true;

    // load reference matricss
    auto hQ   = load_fp16("../matrices/batch_massive_causal/Q.f32", qk_elems);
    auto hK   = load_fp16("../matrices/batch_massive_causal/K.f32", qk_elems);
    auto hV   = load_fp16("../matrices/batch_massive_causal/V.f32", vo_elems);
    auto hRef = load("../matrices/batch_massive_causal/O.f32", vo_elems);

    // allocate device mem
    // fp16

    __half *dQ, *dK, *dV;
    float *dO;
    cudaMalloc(&dQ, qk_elems * sizeof(__half));
    cudaMalloc(&dK, qk_elems * sizeof(__half));
    cudaMalloc(&dV, vo_elems * sizeof(__half));
    cudaMalloc(&dO, vo_elems * sizeof(float));

    cudaMemcpy(dQ, hQ.data(), qk_elems * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), qk_elems * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), vo_elems * sizeof(__half), cudaMemcpyHostToDevice);

    // float *dQ, *dK, *dV, *dO;
    // cudaMalloc(&dQ, qk_elems * sizeof(float));
    // cudaMalloc(&dK, qk_elems * sizeof(float));
    // cudaMalloc(&dV, vo_elems * sizeof(float));
    // cudaMalloc(&dO, vo_elems * sizeof(float));

    // cudaMemcpy(dQ, hQ.data(), qk_elems * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dK, hK.data(), qk_elems * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dV, hV.data(), vo_elems * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(dO, 0, vo_elems * sizeof(float));

    // launch kernel with autotuner - B_r = 128. B_c = 16 was the best result: 2nd fastest time but doesnt put smem at capacity compared to B_c = 32
    // those were best B_r, B_c, but smem is at the max amount 
    printf("B_r,B_c,ms,max_err,smem_bytes\n");
    // tuner<h,  16,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  16,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  16,  64>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  32,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  32,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  32,  64>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  64,  16>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  64,  32>(dQ, dK, dV, dO, hRef, b, N);
    // tuner<h,  64,  64>(dQ, dK, dV, dO, hRef, b, N);
    tuner<h, 128, 16, d_k, d_v, causal_masking>(dQ, dK, dV, dO, hRef, b, N);
    tuner<h, 128, 32, d_k, d_v, causal_masking>(dQ, dK, dV, dO, hRef, b, N);

    const int b_r = 128;
    const int b_c = 32;

    // x = query blocks, y = heads, z = batches
    dim3 grid(CEIL_DIV(N, b_r), h, b);

    // launch kernel with template dims
    flash_attention<h, b_r, b_c, d_k, d_v, causal_masking><<<grid, 2 * b_r>>>(dQ, dK, dV, dO, N);

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