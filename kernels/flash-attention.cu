#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <vector>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

using namespace std;

// flash attention kernel for a single tile, but laid some ground work for multiple tiles and blocks
// B_r - query rows a single block owns, B_c keys per tile
template <int B_r, int B_c, int D_K, int D_V>
__global__ void flash_attention(float *Q, float *K, float *V, float *O, int N) {
    int T_c = CEIL_DIV(N, B_c);                                                                                                                                                                                                                             

    // one thread per row (unnecessary for one block, but keeping for increased size)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread calculates one row of S, O
    const int tm = 1;
    const int tn_s = B_c;
    const int tn_o = D_V;

    // smem registers
    __shared__ float Qs[B_r * D_K];
    __shared__ float Ks[B_c * D_K];
    __shared__ float Vs[B_c * D_V];

    // matmul accumulators
    float S[tm * tn_s];
    float O_acc[tm * tn_o] = {0.0f};
    float m = -INFINITY;
    float ell = 0;

    float scale = rsqrtf((float)D_K);

    // ensures that no extra threads are active if N / B_r isn't a whole number
    // allows for variety of dimensions
    bool active = tid < N;

    // load Q into sram - each thread loads in a column
    // tile is N for now so load full matrix

    // loads are now indexed by block-local threads
    for (int idx = threadIdx.x; idx < B_r * D_K; idx += blockDim.x) {
        // multi block indexing
        int row = idx / D_K;
        int col = idx % D_K;

        // row = curr block * block row size + row of loop
        int g_row = blockIdx.x * B_r + row;

        if (g_row < N) { 
            Qs[idx] = Q[g_row * D_K + col];
        }
    }

    __syncthreads();


    for (int iter = 0; iter < T_c; ++iter) {
        // load K and V into sram - per thread
        // coalesced - each consecutive thread accesses adjacent elements per iter
        for (int idx = threadIdx.x; idx < B_c * D_K; idx += blockDim.x) {
            int col = idx % D_K;
            int row = idx / D_K;

            // global row
            int g_row = iter * B_c + row;

            // don't go beyond dims - not whole division
            if (g_row < N) {
                Ks[row * D_K + col] = K[g_row * D_K + col]; 
            }
        }

        for (int idx = threadIdx.x; idx < B_c * D_V; idx += blockDim.x) {
            int col = idx % D_V;
            int row = idx / D_V;

            // global row
            int g_row = iter * B_c + row;

            // don't go beyond dims
            if (g_row < N) {
                Vs[row * D_V + col] = V[g_row * D_V + col]; 
            }
        }

        __syncthreads();

        // guarding from inactive threads
        if (active) {
            // doesn't overwrite tile row 2 on last iteration, so we can't iterate over it during S calculation
            int cols = min(B_c, N - iter * B_c);

            // S matrix calculation - Q @ K.T
            for (int col = 0; col < cols; ++col) {
                float acc = 0.0f;
                
                for (int k = 0; k < D_K; ++k) {
                    // Qs indexed by local tile - mutliple blocks
                    acc += Qs[threadIdx.x * D_K + k] * Ks[col * D_K + k];
                }

                S[col] = acc * scale;
            }

            float m_j = -INFINITY;

            // find rowmax of tile
            for (int i = 0; i < cols; ++i) {
                m_j = fmaxf(m_j, S[i]);
            }

            // corrections
            float m_new = fmaxf(m, m_j);
            float corr = expf(m - m_new);
            ell = corr * ell;

            for (int v = 0; v < D_V; ++v) {
                // per element correction of O
                O_acc[v] *= corr;
            }

            // compute O - only calc P per element, no need for storage
            for (int col = 0; col < cols; ++col) {
                float P = expf(S[col] - m_new);

                // rowsum of exponentials
                ell += P;

                for (int v = 0; v < D_V; ++v) {
                    O_acc[v] += P * Vs[col * D_V + v];
                }
            }

            // correct max
            m = m_new;
        }

        __syncthreads();
    }
     
    if (active) {
        // normalization and write - per thread
        for (int i = 0; i < D_V; ++i) {
            O_acc[i] = O_acc[i] / ell;
            O[tid * D_V + i] = O_acc[i];
        }
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
    auto hQ = load("../matrices/massive/Q.f32", N * d_k);
    auto hK = load("../matrices/massive/K.f32", N * d_k);
    auto hV = load("../matrices/massive/V.f32", N * d_v);
    auto hRef = load("../matrices/massive/O.f32", N * d_v);

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
    const int b_r = 64;
    const int b_c = 32;

    // amt of blocks = ceil(Q rows / Q rows per block)
    int blocks = CEIL_DIV(N, b_r);

    // launch kernel with template dims
    flash_attention<b_r, b_c, d_k, d_v><<<blocks, b_r>>>(dQ, dK, dV, dO, N);

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
