#include <vector>
#include <cuda_runtime.h>

#include "tuner.cuh"
#include "flash-attention.cuh"

// load reference matrices into vectors
vector<float> load(const char *path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    fread(v.data(), sizeof(float), n, f);
    fclose(f);
    return v;
}

// checker
void checker(vector<float> O, vector<float> O_ref, int N, int d_v) {
    float max_abs = 0.0f;
    int bad = -1;

    // indexes first miscalculation
    for (int i = 0; i < N * d_v; i++) {
        float d = fabsf(O[i] - O_ref[i]);
        if (d > max_abs) max_abs = d;
        if (bad < 0 && d > 1e-5f) bad = i;
    }

    printf("max abs err %.3e\n", max_abs);
    if (bad >= 0)
        printf("FAIL: first at [%d, %d] got %.8f want %.8f\n",
               bad / d_v, bad % d_v, O[bad], O_ref[bad]);
    else
        printf("PASS\n");

    // for (int r = 0; r < N; r++) {
    //     printf("row %d  got:", r);
    //     for (int c = 0; c < d_v; c++) printf(" %8.5f", O[r * d_v + c]);
    //     printf("   want:");
    //     for (int c = 0; c < d_v; c++) printf(" %8.5f", O_ref[r * d_v + c]);
    //     printf("\n");
    // }
}


int main() {
    const int N = 4096, d_k = 64, d_v = 64;

    // load reference matricss
    auto hQ = load("../matrices/massive_causal/Q.f32", N * d_k);
    auto hK = load("../matrices/massive_causal/K.f32", N * d_k);
    auto hV = load("../matrices/massive_causal/V.f32", N * d_v);
    auto hRef = load("../matrices/massive_causal/O.f32", N * d_v);

    // allocate device mem
    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, N * d_k * sizeof(float));
    cudaMalloc(&dK, N * d_k * sizeof(float));
    cudaMalloc(&dV, N * d_v * sizeof(float));
    cudaMalloc(&dO, N * d_v * sizeof(float));

    cudaMemcpy(dQ, hQ.data(), N * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), N * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), N * d_v * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dO, 0, N * d_v * sizeof(float));

    // launch kernel with autotuner - B_r = 64. B_c = 32 was the best result
    // printf("B_r,B_c,ms,max_err,smem_bytes\n");
    // bench<16,  16>(dQ, dK, dV, dO, hRef, N);
    // bench<16,  32>(dQ, dK, dV, dO, hRef, N);
    // bench<16,  64>(dQ, dK, dV, dO, hRef, N);
    // bench<32,  16>(dQ, dK, dV, dO, hRef, N);
    // bench<32,  32>(dQ, dK, dV, dO, hRef, N);
    // bench<32,  64>(dQ, dK, dV, dO, hRef, N);
    // bench<64,  16>(dQ, dK, dV, dO, hRef, N);
    // bench<64,  32>(dQ, dK, dV, dO, hRef, N);
    // bench<64,  64>(dQ, dK, dV, dO, hRef, N);
    // bench<128, 16>(dQ, dK, dV, dO, hRef, N);
    // bench<128, 32>(dQ, dK, dV, dO, hRef, N);
    // bench<128, 64>(dQ, dK, dV, dO, hRef, N);

    const int b_r = 64;
    const int b_c = 32;

    // amt of blocks = ceil(Q rows / Q rows per block)
    int blocks = CEIL_DIV(N, b_r);

    // launch kernel with template dims
    flash_attention<b_r, b_c, d_k, d_v, true><<<blocks, b_r>>>(dQ, dK, dV, dO, N);

    cudaGetLastError();
    cudaDeviceSynchronize();

    // allocate + copy output matrix to host
    std::vector<float> hO(N * d_v);
    cudaMemcpy(hO.data(), dO, N * d_v * sizeof(float), cudaMemcpyDeviceToHost);

    // run checker
    checker(hO, hRef, N, d_v);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    return 0;
}