#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <vector>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define ks_size 32
#define vs_size 48

using namespace std;

// flash attention kernel for a single tile, but laid some ground work for multiple tiles and blocks
__global__ void flash_attention(float *Q, float *K, float *V, float *O, int N, int d_k, int d_v) {
    // tile sizes
    const int B_r = N; // rows
    const int B_c = N; // columns

    // one thread per row (unnecessary for one block, but keeping for increased size)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread calculates one row of S, O
    const int tm = 1;
    const int tn_s = 8;
    const int tn_o = 6;

    // smem registers (only K and V for now)
    __shared__ float Ks[ks_size];
    __shared__ float Vs[vs_size];

    // matmul accumulators
    float S[tm * tn_s];
    float O_acc[tm * tn_o] = {0.0f};

    // int T_r = CEIL_DIV(N, B_r);
    // int T_ck = CEIL_DIV(N, B_c); // transpose so need N as dividend

    float scale = rsqrtf((float)d_k);

    // load Q and K into sram - per tjh
    for (int i = 0; i < d_k; ++i) {
        Ks[tid * d_k + i] = K[tid * d_k + i];
    }

    for (int i = 0; i < d_v; ++i) {
        Vs[tid * d_v + i] = V[tid * d_v + i];
    }

    __syncthreads();

    if (tid >= N) return;

    // S matrix calculation - Q @ K.T
    for (int col = 0; col < B_c; ++col) {
        float acc = 0.0f;
        
        for (int k = 0; k < d_k; ++k) {
            acc += Q[tid * d_k + k] * Ks[col * d_k + k];
        }

        S[col] = acc * scale;
    }

    // one thread owns a single row for this version
    float m = -INFINITY;
    float ell = 0;

    // online softmax (doesn't do much for a single tile, but will be beneficial for multiple)
    for (int i = 0; i < N; ++i) {
        float m_new = fmaxf(m, S[i]);
        ell = expf(m - m_new) * ell + expf(S[i] - m_new);
        m = m_new;
    }

    // compute O - only calc P per element, no need for storage
    for (int col = 0; col < B_c; ++col) {
        float P = expf(S[col] - m);

        for (int v = 0; v < d_v; ++v) {
            O_acc[v] += P * Vs[col * d_v + v];
        }
    }

    // normalization and write - per thread
    for (int i = 0; i < d_v; ++i) {
        O_acc[i] = O_acc[i] / ell;
        O[tid * d_v + i] = O_acc[i];
    }
}

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

    for (int r = 0; r < N; r++) {
        printf("row %d  got:", r);
        for (int c = 0; c < d_v; c++) printf(" %8.5f", O[r * d_v + c]);
        printf("   want:");
        for (int c = 0; c < d_v; c++) printf(" %8.5f", O_ref[r * d_v + c]);
        printf("\n");
    }
}


int main() {
    int N = 8, d_k = 4, d_v = 6;

    // load reference matricss
    auto hQ = load("../matrices/baseline/Q.f32", N * d_k);
    auto hK = load("../matrices/baseline/K.f32", N * d_k);
    auto hV = load("../matrices/baseline/V.f32", N * d_v);
    auto hRef = load("../matrices/baseline/O.f32", N * d_v);

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

    // launch kernel - one block for this first single tile algorithm
    flash_attention<<<1, N>>>(dQ, dK, dV, dO, N, d_k, d_v);
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
