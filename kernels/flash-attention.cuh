#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

// flash attention kernel for a single tile, but laid some ground work for multiple tiles and blocks
// B_r - query rows a single block owns, B_c keys per tile
template <int H, int B_r, int B_c, int D_K, int D_V, bool causal, typename T = float>
__global__ void flash_attention(T *Q, T *K, T *V, float *O, int N) {
    // matrix dims: [Batches (B), Sequence Length (N), Attention Head (H), Dimensionality (D)]                                                                                                                                                                                                                          

    // one thread per row (unnecessary for one block, but keeping for increased size)
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // warp partitioning: two threads per row
    int row_idx = threadIdx.x / 2; 
    int row_half = threadIdx.x % 2;

    // query row idx is the row idx within the block
    int q_idx = blockIdx.x * B_r + row_idx;

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

    // each thread calculates half of a row of S, O
    const int tm = 1;
    const int tn_s = B_c;
    const int tn_o = D_V / 2; // each thread computes the final dp of D_V / 2 elements in O_acc

    // matmul accumulator registers - per thread
    float S[tm * tn_s]; // warp partitioning makes each element a partial sum - needs shuffling
    float O_acc[tm * tn_o] = {0.0f}; // each thrad 
    float m = -CUDART_INF_F;
    float ell = 0;

    // scale val for attention
    // putting into log2 space to enable exp2f - better perf on GPUs
    // expf does the conversion per call, change once here to mitigate that 
    float scale = rsqrtf((float)D_K) * 1.4426950408889634f; 

    // smem registers
    __shared__ __half Qs[B_r * D_K];
    // two arrays for double buffering
    // align to 16 bytes - casting later
    __align__(16) __shared__ __half Ks[2][B_c * D_K];
    __align__(16) __shared__ __half Vs[2][B_c * D_V];

    // ensures that no extra threads are active if N / B_r isn't a whole number
    // allows for variety of dimensions
    bool active = q_idx < N;

    // load Q into sram - each thread loads in a column
    // loads are now indexed by block-local threads
    for (int idx = threadIdx.x; idx < B_r * D_K; idx += blockDim.x) {
        // multi block indexing
        int row = idx / D_K;
        int col = idx % D_K;

        // global row = curr block * block row size + row of loop
        int g_row = blockIdx.x * B_r + row;

        if (g_row < N) { 
            Qs[idx] = __float2half(Q[g_row * stride_n_qk + col]);
        }
    }

    __syncthreads();  

    // prologue - async loads of the first tiles of K, V
    // async loads require 4, 8, or 16 bytes - using 16 here
    // iterating 8 - 8 * 2 = 16 bytes
    for (int idx = threadIdx.x; idx < B_c * (D_K / 8); idx += blockDim.x) {
        // % D_K / 8 gives you the chunk, * 8 is starting column
        // D_K / 8 chunks per row, 8 elements per chunk
        int col = idx % (D_K / 8) * 8;

        // (D_K / 8 chunks per row - width)
        int row = idx / (D_K / 8);

        // global row = local row for first tile (iter = 0)
        int g_row = row;

        // padding for uneven dims and async loads
        // stride_n_qk * g_row is only 16 byte aligned if D_K * H * 2 is a multiple of 16
        if (g_row < N) {
            __pipeline_memcpy_async(reinterpret_cast<float4*>(&Ks[0][row * D_K + col]),
                reinterpret_cast<const float4*>(&K[g_row * stride_n_qk + col]), sizeof(float4));
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                Ks[0][row * D_K + col + i] = __float2half(0.0f);
            }
        }
    }

    // idx indexes a 16 byte (8 element chunk), D_V / 8 chunks
    for (int idx = threadIdx.x; idx < B_c * (D_V / 8); idx += blockDim.x) {
        // % D_V / 8 gives you the chunk, * 8 is starting column
        // D_V / 8 chunks per row, 8 elements per chunk
        int col = idx % (D_V / 8) * 8;

        // (D_V / 8 chunks per row - width)
        int row = idx / (D_V / 8);

        // global row = local row for first tile (iter = 0)
        int g_row = row;

        // padding for uneven dims and async loads
        // float4 - 16 bytes
        if (g_row < N) {
            __pipeline_memcpy_async(reinterpret_cast<float4*>(&Vs[0][row * D_V + col]),
                reinterpret_cast<const float4*>(&V[g_row * stride_n_vo + col]), sizeof(float4));
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                Vs[0][row * D_V + col + i] = __float2half(0.0f);
            }
        }
    }

    __pipeline_commit();

    // tile amount
    int T_c = CEIL_DIV(N, B_c); 

    for (int iter = 0; iter < T_c; ++iter) {
        // load K and V into sram - per thread
        // coalesced - each consecutive thread accesses adjacent elements per iter
        // double buffering - must ensure next batch is within tile amount
        if (iter + 1 < T_c) {
            for (int idx = threadIdx.x; idx < B_c * (D_K / 8); idx += blockDim.x) {
                int col = idx % (D_K / 8) * 8;
                int row = idx / (D_K / 8);

                // global row
                int g_row = (iter + 1) * B_c + row;

                // padding to allow various dim sizes
                // double buffering - 16 bytes, cast to float4
                if (g_row < N) {
                    __pipeline_memcpy_async(reinterpret_cast<float4*>(&Ks[(iter + 1) % 2][row * D_K + col]),
                        reinterpret_cast<const float4*>(&K[g_row * stride_n_qk + col]), sizeof(float4));
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        Ks[(iter + 1) % 2][row * D_K + col + i] = __float2half(0.0f);
                    }   
                }
            }

            for (int idx = threadIdx.x; idx < B_c * (D_V / 8); idx += blockDim.x) {
                int col = idx % (D_V / 8) * 8;
                int row = idx / (D_V / 8);

                // global row
                int g_row = (iter + 1) * B_c + row;

                // padding to allow various dim sizes
                // convert fp32 to fp16 vals
                if (g_row < N) {
                    __pipeline_memcpy_async(reinterpret_cast<float4*>(&Vs[(iter + 1) % 2][row * D_V + col]),
                        reinterpret_cast<const float4*>(&V[g_row * stride_n_vo + col]), sizeof(float4));
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        Vs[(iter + 1) % 2][row * D_V + col + i] = __float2half(0.0f);
                    }
                }
            }

            __pipeline_commit();

            // two commits - only want the oldest one
            __pipeline_wait_prior(1);
        }

        else {
            // no prefetch - only 1 commit left
            __pipeline_wait_prior(0);
        }

        __syncthreads();

        // tile skipping - entire tile is masked if the highest query index is less than the lowest key index
        int lowest_kidx = B_c * iter;
        bool skip = causal && lowest_kidx > highest_qidx; // only skip if kernel is actually using causal masking and condition is met

        // this is safe because threads won't diverge as this is a block-level check
        // need to check skip condition before prefetched loads (will optimize soon) - no point in loading if vals get skipped
        if (skip) continue;

        // guarding from inactive threads
        if (active) {
            // tile might be too short (less cols than the others) - doesn't overwrite
            // int cols = min(B_c, N - iter * B_c);


            // First matmul: S matrix calculation - Q @ K.T
            // warp partitioning: start at the first col of the half the thread is working on, iterate for D_K / 2 (half)
            #pragma unroll
            for (int col = 0; col < B_c; ++col) {
                float acc = 0.0f;
                int k_idx = iter * B_c + col; // key idx is the global key index = tile offset + local col
                
                #pragma unroll
                for (int k = row_half * (D_K / 2); k < row_half * (D_K / 2) + (D_K / 2); ++k) {
                    // Qs indexed by local tile - mutliple blocks
                    // recast fp16 vals to fp32 to minimize round errors
                    // double buffering: needs iter % 2 index
                    acc += __half2float(Qs[row_idx * D_K + k]) * __half2float(Ks[iter % 2][col * D_K + k]);
                }

                S[col] = acc * scale;

                // masking: key col > query row - sets element to negative infinity
                // handles various dim sizes for N - some tiles may only have x < B_c real columns, so only put values in those
                if (k_idx >= N || (causal && k_idx > q_idx)) {
                    S[col] = -CUDART_INF_F;
                }
            }

            // shuffle
            // each entry needs its own shuffle: two partials across two lanes, B_c entries, B_c shuffles
            #pragma unroll
            for (int col = 0; col < B_c; ++col) {
                S[col] += __shfl_xor_sync(__activemask(), S[col], 1);
            }
            
            float m_j = -CUDART_INF_F;

            // find rowmax of tile
            // can now use B_c since padding is implemented - extra cols won't have an impact
            #pragma unroll
            for (int i = 0; i < B_c; ++i) {
                m_j = fmaxf(m_j, S[i]);
            }

            // corrections
            float m_new = fmaxf(m, m_j);
            // masking
            bool masked_out = (m_new == -CUDART_INF_F);

            // only runs if some part of row isn't masked - doesn't need to run if row section is completely masked
            if (!masked_out) {
                float corr = exp2f(m - m_new);
                ell = corr * ell;
                
                // O_acc is only D_V / 2 wide now
                for (int v = 0; v < D_V / 2; ++v) {
                    // per element correction of O
                    O_acc[v] *= corr;
                }
            

                // Second matmul: compute O - only calc P per element, no need for storage
                // can now use B_c since padding is implemented - extra cols won't have an impact
                // warp partitioning: each thread does half of its row
                #pragma unroll
                for (int col = 0; col < B_c; ++col) {
                    float P = exp2f(S[col] - m_new);

                    // rowsum of exponentials
                    ell += P;

                    for (int v = 0; v < D_V / 2; ++v) {
                        // col * D_V gets row, row_half * D_V / 2 gets starting half, v gets correct col within that half
                        // double buffering: needs iter % 2 index
                        O_acc[v] += P * __half2float(Vs[iter % 2][col * D_V + row_half * (D_V / 2) + v]);
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
        for (int i = 0; i < D_V / 2; ++i) {
            O_acc[i] = O_acc[i] / ell;
            // q_idx * stride_n_vo gets row, row_half * D_V / 2 gets starting half, i gets correct col within that half
            O[q_idx * stride_n_vo + row_half * (D_V / 2) + i] = O_acc[i];
        }
    }
}