#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

// flash attention kernel for a single tile, but laid some ground work for multiple tiles and blocks
// B_r - query rows a single block owns, B_c keys per tile
template <int H, int B_r, int B_c, int D_K, int D_V, bool causal>
__global__ void flash_attention(float *Q, float *K, float *V, float *O, int N) {
    // matrix dims: [Batches (B), Sequence Length (N), Attention Head (H), Dimensionality (D)]                                                                                                                                                                                                                          

    // one thread per row (unnecessary for one block, but keeping for increased size)
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // query row idx is the thread id within the block
    int q_idx = blockIdx.x * B_r + threadIdx.x;
    // highest query row idx within block
    int highest_qidx = blockIdx.x * B_r + B_r - 1;

    // block and head dims correspond to y, z
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    // pointer offsets to locate correct batch and head within Q, K, V, O
    const size_t qk_off = ((size_t) b * (N * H * D_K)) + h * D_K;
    const size_t vo_off = ((size_t) b * (N * H * D_V)) + h * D_V;

    // size of each (H, D) slice within N
    const int stride_n_qk = H * D_K;
    const int stride_n_vo = H * D_V;

    // shift pointers
    Q += qk_off;
    K += qk_off;
    V += vo_off;
    O += vo_off;

    // each thread calculates one row of S, O
    const int tm = 1;
    const int tn_s = B_c;
    const int tn_o = D_V;

    // matmul accumulators - per thread
    float S[tm * tn_s];
    float O_acc[tm * tn_o] = {0.0f};
    float m = -CUDART_INF_F;
    float ell = 0;

    float scale = rsqrtf((float)D_K); // scale val for attention

    // smem registers
    __shared__ float Qs[B_r * D_K];
    __shared__ float Ks[B_c * D_K];
    __shared__ float Vs[B_c * D_V];

    // ensures that no extra threads are active if N / B_r isn't a whole number
    // allows for variety of dimensions
    bool active = q_idx < N;

    // load Q into sram - each thread loads in a column
    // loads are now indexed by block-local threads
    for (int idx = threadIdx.x; idx < B_r * D_K; idx += blockDim.x) {
        // multi block indexing
        int row = idx / D_K;
        int col = idx % D_K;

        // row = curr block * block row size + row of loop
        int g_row = blockIdx.x * B_r + row;

        if (g_row < N) { 
            Qs[idx] = Q[g_row * stride_n_qk + col];
        }
    }

    __syncthreads();

    // tile amount
    int T_c = CEIL_DIV(N, B_c);   

    for (int iter = 0; iter < T_c; ++iter) {

        // tile skipping - entire tile is masked if the highest query index is less than the lowest key index
        int lowest_kidx = B_c * iter;
        bool skip = causal && lowest_kidx > highest_qidx; // only skip if kernel is actually using causal masking and condition is met

        // this is safe because threads won't diverge as this is a block-level check
        if (skip) continue;


        // load K and V into sram - per thread
        // coalesced - each consecutive thread accesses adjacent elements per iter
        for (int idx = threadIdx.x; idx < B_c * D_K; idx += blockDim.x) {
            int col = idx % D_K;
            int row = idx / D_K;

            // global row
            int g_row = iter * B_c + row;

            // don't go beyond dims - not whole division
            if (g_row < N) {
                Ks[row * D_K + col] = K[g_row * stride_n_qk + col]; 
            }
        }

        for (int idx = threadIdx.x; idx < B_c * D_V; idx += blockDim.x) {
            int col = idx % D_V;
            int row = idx / D_V;

            // global row
            int g_row = iter * B_c + row;

            // don't go beyond dims
            if (g_row < N) {
                Vs[row * D_V + col] = V[g_row * stride_n_vo + col]; 
            }
        }

        __syncthreads();

        // guarding from inactive threads
        if (active) {
            // tile might be too short (less cols than the others) - doesn't overwrite
            int cols = min(B_c, N - iter * B_c);

            // S matrix calculation - Q @ K.T
            for (int col = 0; col < cols; ++col) {
                float acc = 0.0f;
                int k_idx = iter * B_c + col; // key idx is the global key index = tile offset + local col
                
                for (int k = 0; k < D_K; ++k) {
                    // Qs indexed by local tile - mutliple blocks
                    acc += Qs[threadIdx.x * D_K + k] * Ks[col * D_K + k];
                }

                S[col] = acc * scale;

                // masking: key col > query row
                if (causal && k_idx > q_idx) {
                    S[col] = -CUDART_INF_F;
                }
            }

            float m_j = -CUDART_INF_F;

            // find rowmax of tile
            for (int i = 0; i < cols; ++i) {
                m_j = fmaxf(m_j, S[i]);
            }

            // corrections
            float m_new = fmaxf(m, m_j);
            // masking
            bool masked_out = (m_new == -CUDART_INF_F);

            // only runs if some part of row isn't masked - doesn't need to run if row section is completely masked
            if (!masked_out) {
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
            O[q_idx * stride_n_vo + i] = O_acc[i];
        }
    }
}