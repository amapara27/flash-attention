#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <vector>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

using namespace std;

// flash attention kernel - one block, but laid some ground work for multiple tiles and blocks
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

    if (tid >= N) return;

    // matmul accumulators
    float S[tm * tn_s];
    float O_acc[tm * tn_o] = {0.0f};

    // int T_r = CEIL_DIV(N, B_r);
    // int T_ck = CEIL_DIV(N, B_c); // transpose so need N as dividend

    float scale = rsqrtf((float)d_k);

    // S matrix calculation - Q @ K.T
    for (int col = 0; col < B_c; col++) {
        float acc = 0.0f;
        
        for (int k = 0; k < B_c; k++) {
            acc += Q[tid * d_k + k] * K[col * d_k + k];
        }

        S[col] = acc * scale;
    }
}

// load reference matrices into vectors
std::vector<float> load(const char *path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    fread(v.data(), sizeof(float), n, f);
    fclose(f);
    return v;
}


int main() {
    int N = 8, d_k = 4, d_v = 6;
    auto Q = load("../matrices/baseline/Q.f32", N * d_k);
    auto K = load("../matrices/baseline/Q.f32", N * d_k);
    auto V = load("../matrices/baseline/Q.f32", N * d_v);
    auto O_ref = load("../matrices/baseline/Q.f32", N * d_v);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < d_k; j++) {
            cout << Q[i * d_k + j];
        }
        cout << "\n";
    }

    return 0;
}
