#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <vector>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

using namespace std;

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
