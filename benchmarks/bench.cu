// Unified benchmark harness - one binary per stage.
//   build: nvcc -O3 -arch=sm_86 -lineinfo -DSTAGE=<n> bench.cu -o bin/bench_s<n>.exe   (STAGE 91 = 09b)
//   run:   bench_sNN --fixture <dir> --causal 0|1 --br <B_r> --bc <B_c> --mode check|profile|time
//
// modes
//   check   - 1 launch, compare against reference
//   profile - 1 warmup + 5 launches (pair with ncu --launch-skip 1 --launch-count 5), then compare
//   time    - 3 warmup + 50 cudaEvent-timed launches, median, then compare
//
// output: one CSV line
//   stage,fixture,causal,dtype,br,bc,threads,smem_bytes,max_err,max_err_vs_fp32,status,wall_ms

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

// ---------------------------------------------------------------- stage selection
#if   STAGE == 1
  #include "stages/01_tiled/kernel.cuh"
  #define STAGE_NAME "01_tiled"
#elif STAGE == 2
  #include "stages/02_causal_mask/kernel.cuh"
  #define STAGE_NAME "02_causal_mask"
#elif STAGE == 3
  #include "stages/03_tile_skipping/kernel.cuh"
  #define STAGE_NAME "03_tile_skipping"
#elif STAGE == 4
  #include "stages/04_multihead/kernel.cuh"
  #define STAGE_NAME "04_multihead"
#elif STAGE == 5
  #include "stages/05_s_registers/kernel.cuh"
  #define STAGE_NAME "05_s_registers"
#elif STAGE == 6
  #include "stages/06_warp_partitioning/kernel.cuh"
  #define STAGE_NAME "06_warp_partitioning"
#elif STAGE == 7
  #include "stages/07_fp16_smem/kernel.cuh"
  #define STAGE_NAME "07_fp16_smem"
#elif STAGE == 8
  #include "stages/08_double_buffering/kernel.cuh"
  #define STAGE_NAME "08_double_buffering"
#elif STAGE == 9
  #include "stages/09_wmma_qk/kernel.cuh"
  #define STAGE_NAME "09_wmma_qk"
#elif STAGE == 91
  #include "stages/09b_wmma_qk_fixed/kernel.cuh"
  #define STAGE_NAME "09b_wmma_qk_fixed"
#else
  #error "unknown STAGE"
#endif

constexpr bool MULTIHEAD = STAGE >= 4;
constexpr bool FP16_IN   = STAGE >= 7;
constexpr int  ROW_THREADS = STAGE >= 6 ? 2 : 1;   // threads per query row
using T_in = std::conditional_t<FP16_IN, __half, float>;

// static smem per block, mirrors each stage's __shared__ declarations
template <int BR, int BC, int DK, int DV>
constexpr int smem_bytes() {
    if constexpr (STAGE <= 6) return 4 * (BR * DK + BC * DK + BC * DV);
    else if constexpr (STAGE == 7) return 2 * (BR * DK + BC * DK + BC * DV);
    else if constexpr (STAGE == 8) return 2 * (BR * DK + 2 * BC * DK + 2 * BC * DV);
    else return 2 * (BR * DK + 2 * BC * DK + 2 * BC * DV) + 4 * BR * BC;
}

template <int BR, int BC, int DK, int DV>
constexpr bool valid_config() {
    if (smem_bytes<BR, BC, DK, DV>() > 48 * 1024) return false;
    if (STAGE == 9 && BC != 32) return false;   // committed wmma kernel zeroes s_frag[0..1] only
    return true;
}

// ---------------------------------------------------------------- per-stage launch
template <int H, int BR, int BC, int DK, int DV, bool CAUSAL>
void run_kernel(T_in *Q, T_in *K, T_in *V, float *O, int B, int N) {
    const int threads = ROW_THREADS * BR;
#if STAGE == 1
    static_assert(!CAUSAL && H == 1, "stage 1 is single-head, non-causal");
    flash_attention<BR, BC, DK, DV><<<CEIL_DIV(N, BR), threads>>>(Q, K, V, O, N);
#elif STAGE <= 3
    static_assert(H == 1, "stages 2-3 are single-head");
    flash_attention<BR, BC, DK, DV, CAUSAL><<<CEIL_DIV(N, BR), threads>>>(Q, K, V, O, N);
#elif STAGE <= 7
    flash_attention<H, BR, BC, DK, DV, CAUSAL><<<dim3(CEIL_DIV(N, BR), H, B), threads>>>(Q, K, V, O, N);
#elif STAGE == 8
    flash_attention<H, BR, BC, DK, DV, CAUSAL, __half><<<dim3(CEIL_DIV(N, BR), H, B), threads>>>(Q, K, V, O, N);
#else
    flash_attention_wmma<H, BR, BC, DK, DV, CAUSAL, __half><<<dim3(CEIL_DIV(N, BR), H, B), threads>>>(Q, K, V, O, N);
#endif
}

struct Launch {
    void (*fn)(T_in *, T_in *, T_in *, float *, int, int) = nullptr;
    int threads = 0, smem = 0;
};

template <int H, int DV, bool CAUSAL>
Launch pick_tile(int br, int bc) {
    Launch l;
#define TRY(R, C)                                                              \
    if (br == R && bc == C) {                                                  \
        if constexpr (valid_config<R, C, 64, DV>()) {                          \
            l.fn = run_kernel<H, R, C, 64, DV, CAUSAL>;                        \
            l.threads = ROW_THREADS * R;                                       \
            l.smem = smem_bytes<R, C, 64, DV>();                               \
        }                                                                      \
        return l;                                                              \
    }
    TRY(16, 16) TRY(16, 32) TRY(16, 64)
    TRY(32, 16) TRY(32, 32) TRY(32, 64)
    TRY(64, 16) TRY(64, 32) TRY(64, 64)
    TRY(128, 16) TRY(128, 32) TRY(128, 64)
#undef TRY
    return l;
}

template <int H, int DV>
Launch pick_causal(bool causal, int br, int bc) {
    if (causal) {
        if constexpr (STAGE == 1) return Launch{};
        else return pick_tile<H, DV, true>(br, bc);
    }
    return pick_tile<H, DV, false>(br, bc);
}

// supported (H, D_V) shapes: the perf fixtures, plus the ragged correctness fixtures with -DODD_SHAPES
Launch pick(int H, int DV, bool causal, int br, int bc) {
    if (H == 1 && DV == 64) return pick_causal<1, 64>(causal, br, bc);
    if constexpr (MULTIHEAD) {
        if (H == 4 && DV == 64) return pick_causal<4, 64>(causal, br, bc);
    }
#ifdef ODD_SHAPES   // large_odd, batch_odd - opt-in, roughly doubles compile time
    if (H == 1 && DV == 48) return pick_causal<1, 48>(causal, br, bc);
    if constexpr (MULTIHEAD) {
        if (H == 3 && DV == 48) return pick_causal<3, 48>(causal, br, bc);
    }
#endif
    return Launch{};
}

// ---------------------------------------------------------------- host helpers
static std::vector<float> load_f32(const std::string &p, size_t n) {
    std::vector<float> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p.c_str()); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "FATAL: %s has %zu floats, want %zu\n", p.c_str(), got, n); exit(1); }
    return v;
}

static T_in *upload(const std::vector<float> &h) {
    std::vector<T_in> conv(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        if constexpr (FP16_IN) conv[i] = __float2half(h[i]);
        else conv[i] = h[i];
    }
    T_in *d;
    cudaMalloc(&d, conv.size() * sizeof(T_in));
    cudaMemcpy(d, conv.data(), conv.size() * sizeof(T_in), cudaMemcpyHostToDevice);
    return d;
}

// max abs error; NaN / unwritten outputs count as infinite error
static double max_err(const std::vector<float> &O, const std::vector<float> &ref) {
    double m = 0.0;
    for (size_t i = 0; i < O.size(); ++i) {
        if (!std::isfinite(O[i])) return std::numeric_limits<double>::infinity();
        m = std::max(m, (double)std::fabs(O[i] - ref[i]));
    }
    return m;
}

static const char *arg(int argc, char **argv, const char *key, const char *def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

int main(int argc, char **argv) {
    std::string fixture = arg(argc, argv, "--fixture", "fixtures/single_4096");
    bool causal = atoi(arg(argc, argv, "--causal", "1")) != 0;
    int br = atoi(arg(argc, argv, "--br", "64"));
    int bc = atoi(arg(argc, argv, "--bc", "32"));
    std::string mode = arg(argc, argv, "--mode", "check");
    double tol = atof(arg(argc, argv, "--tol", "1e-4"));

    int B, N, H, DK, DV;
    FILE *sf = fopen((fixture + "/shape.txt").c_str(), "r");
    if (!sf || fscanf(sf, "%d %d %d %d %d", &B, &N, &H, &DK, &DV) != 5) { fprintf(stderr, "FATAL: bad shape.txt in %s\n", fixture.c_str()); return 1; }
    fclose(sf);

    std::string fx_name = fixture.substr(fixture.find_last_of("/\\") + 1);
    const char *dtype = FP16_IN ? "fp16" : "fp32";

    Launch l = (DK == 64 && (MULTIHEAD || B == 1)) ? pick(H, DV, causal, br, bc) : Launch{};
    if (!l.fn) {
        printf("%s,%s,%d,%s,%d,%d,,,,,UNSUPPORTED,\n", STAGE_NAME, fx_name.c_str(), causal, dtype, br, bc);
        return 3;
    }

    const size_t nqk = (size_t)B * N * H * DK, nvo = (size_t)B * N * H * DV;
    const char *tag = causal ? "c" : "nc";
    auto ref   = load_f32(fixture + "/O_" + tag + (FP16_IN ? "_f16in.f32" : "_f32in.f32"), nvo);
    auto truth = load_f32(fixture + "/O_" + tag + "_f32in.f32", nvo);

    T_in *dQ = upload(load_f32(fixture + "/Q.f32", nqk));
    T_in *dK = upload(load_f32(fixture + "/K.f32", nqk));
    T_in *dV = upload(load_f32(fixture + "/V.f32", nvo));
    float *dO;
    cudaMalloc(&dO, nvo * sizeof(float));
    cudaMemset(dO, 0xFF, nvo * sizeof(float));   // NaN fill: unwritten outputs fail the check

    double wall_ms = -1.0;
    if (mode == "check") {
        l.fn(dQ, dK, dV, dO, B, N);
    } else if (mode == "profile") {
        for (int r = 0; r < 6; ++r) l.fn(dQ, dK, dV, dO, B, N);
    } else if (mode == "time") {
        for (int r = 0; r < 3; ++r) l.fn(dQ, dK, dV, dO, B, N);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        std::vector<float> times;
        for (int r = 0; r < 50; ++r) {
            cudaEventRecord(start);
            l.fn(dQ, dK, dV, dO, B, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        wall_ms = times[times.size() / 2];
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        fprintf(stderr, "FATAL: unknown mode %s\n", mode.c_str());
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("%s,%s,%d,%s,%d,%d,%d,%d,,,LAUNCHFAIL:%s,\n", STAGE_NAME, fx_name.c_str(), causal, dtype,
               br, bc, l.threads, l.smem, cudaGetErrorString(err));
        return 4;
    }

    std::vector<float> hO(nvo);
    cudaMemcpy(hO.data(), dO, nvo * sizeof(float), cudaMemcpyDeviceToHost);
    double e = max_err(hO, ref), e_truth = max_err(hO, truth);
    bool pass = e <= tol;

    printf("%s,%s,%d,%s,%d,%d,%d,%d,%.3e,%.3e,%s,", STAGE_NAME, fx_name.c_str(), causal, dtype,
           br, bc, l.threads, l.smem, e, e_truth, pass ? "PASS" : "FAIL");
    if (wall_ms >= 0) printf("%.4f", wall_ms);
    printf("\n");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return pass ? 0 : 2;
}
