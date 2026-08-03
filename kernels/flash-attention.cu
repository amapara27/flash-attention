#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <vector>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define ks_size 12
#define vs_size 18

using namespace std;

// flash attention kernel for a single tile, but laid some ground work for multiple tiles and blocks
__global__ void flash_attention(float *Q, float *K, float *V, float *O, int N, int d_k, int d_v) {
    // tile sizes
    const int B_r = N; // query rows this block owns
    const int B_c = 3; // keys per iteration (tiling)

    int T_c = CEIL_DIV(N, B_c);                                                                                                                                                                                                                             

    // one thread per row (unnecessary for one block, but keeping for increased size)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread calculates one row of S, O
    const int tm = 1;
    const int tn_s = B_c;
    const int tn_o = 6;

    // coalescing + loading K, V into smem - unused for now since division of dims is uneven
    int iColK = threadIdx.x % d_k;
    int iRowK = threadIdx.x / d_k;
    int strideK = blockDim.x / d_k;

    int iColV = threadIdx.x % d_v;
    int iRowV = threadIdx.x / d_v;
    int strideV = blockDim.x / d_k;

    // smem registers (only K and V for now)
    __shared__ float Ks[ks_size];
    __shared__ float Vs[vs_size];

    // matmul accumulators
    float S[tm * tn_s];
    float O_acc[tm * tn_o] = {0.0f};
    float m = -INFINITY;
    float ell = 0;

    float scale = rsqrtf((float)d_k);

    if (tid >= N) return;

    for (int iter = 0; iter < T_c; ++iter) {
        // load K and V into sram - per thread
        // coalesced - each consecutive thread accesses adjacent elements per iter
        for (int idx = tid; idx < B_c * d_k; idx += blockDim.x) {
            int col = idx % d_k;
            int row = idx / d_k;

            // global row
            int g_row = iter * B_c + row;

            // don't go beyond dims
            if (g_row < N) {
                Ks[row * d_k + col] = K[(iter * B_c + row) * d_k + col]; 
            }
        }

        for (int idx = tid; idx < B_c * d_v; idx += blockDim.x) {
            int col = idx % d_v;
            int row = idx / d_v;

            // global row
            int g_row = iter * B_c + row;

            // don't go beyond dims
            if (g_row < N) {
                Vs[row * d_v + col] = V[g_row * d_v + col]; 
            }
        }

        __syncthreads();

        // doesn't overwrite tile row 2 on last iteration, so we can't iterate over it during S calculation
        int cols = min(B_c, N - iter * B_c);

        // S matrix calculation - Q @ K.T
        for (int col = 0; col < cols; ++col) {
            float acc = 0.0f;
            
            for (int k = 0; k < d_k; ++k) {
                acc += Q[tid * d_k + k] * Ks[col * d_k + k];
            }

            S[col] = acc * scale;
        }

        __syncthreads();

        float m_j = -INFINITY;

        // find rowmax of tile
        for (int i = 0; i < cols; ++i) {
            m_j = fmaxf(m_j, S[i]);
        }

        // corrections
        float m_new = fmaxf(m, m_j);
        float corr = expf(m - m_new);
        ell = corr * ell;

        for (int v = 0; v < d_v; ++v) {
            // per element correction of O
            O_acc[v] *= corr;
        }

        // compute O - only calc P per element, no need for storage
        for (int col = 0; col < cols; ++col) {
            float P = expf(S[col] - m_new);

            // rowsum of exponentials
            ell += P;

            for (int v = 0; v < d_v; ++v) {
                O_acc[v] += P * Vs[col * d_v + v];
            }
        }

        // correct max
        m = m_new;
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
