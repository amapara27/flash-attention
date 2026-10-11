// fp64 CPU reference attention for the benchmark fixtures.
// For each fixture dir writes four references:
//   O_{nc,c}_f32in.f32 - inputs as stored (fp32)
//   O_{nc,c}_f16in.f32 - inputs rounded to fp16 first (exactly what the fp16 kernels see)
// and compares the matching one against O_original.f32 from reference/dump_trace.py.
// usage: make_refs <fixture_dir> [<fixture_dir> ...]

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

using std::vector; using std::string;

static vector<float> load(const string &p, size_t n) {
    vector<float> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { printf("FATAL: cannot open %s\n", p.c_str()); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { printf("FATAL: %s has %zu floats, want %zu\n", p.c_str(), got, n); exit(1); }
    return v;
}

static void save(const string &p, const vector<float> &v) {
    FILE *f = fopen(p.c_str(), "wb");
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

// layout [B, N, H, D]
static vector<float> attention(const vector<float> &Q, const vector<float> &K, const vector<float> &V,
                               int B, int N, int H, int DK, int DV, bool causal) {
    vector<float> O((size_t)B * N * H * DV);
    const double scale = 1.0 / std::sqrt((double)DK);

    // flattened (b, h, i) loop - MSVC OpenMP doesn't support collapse
    #pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < B * H * N; ++r) {
        {
            int bh = r / N, i = r % N;
            int b = bh / H, h = bh % H;
            vector<double> s(N);
            const float *q = &Q[(((size_t)b * N + i) * H + h) * DK];
            int last = causal ? i : N - 1;
            double mx = -INFINITY;
            for (int j = 0; j <= last; ++j) {
                const float *k = &K[(((size_t)b * N + j) * H + h) * DK];
                double acc = 0.0;
                for (int d = 0; d < DK; ++d) acc += (double)q[d] * (double)k[d];
                s[j] = acc * scale;
                mx = std::max(mx, s[j]);
            }
            double sum = 0.0;
            for (int j = 0; j <= last; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            float *o = &O[(((size_t)b * N + i) * H + h) * DV];
            for (int d = 0; d < DV; ++d) {
                double acc = 0.0;
                for (int j = 0; j <= last; ++j)
                    acc += s[j] * (double)V[(((size_t)b * N + j) * H + h) * DV + d];
                o[d] = (float)(acc / sum);
            }
        }
    }
    return O;
}

static vector<float> round_fp16(const vector<float> &x) {
    vector<float> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = __half2float(__float2half(x[i]));
    return r;
}

static double max_abs_diff(const vector<float> &a, const vector<float> &b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char **argv) {
    for (int a = 1; a < argc; ++a) {
        string dir = argv[a];
        int B, N, H, DK, DV;
        FILE *sf = fopen((dir + "/shape.txt").c_str(), "r");
        if (!sf || fscanf(sf, "%d %d %d %d %d", &B, &N, &H, &DK, &DV) != 5) { printf("FATAL: bad shape.txt in %s\n", dir.c_str()); return 1; }
        fclose(sf);

        size_t nqk = (size_t)B * N * H * DK, nvo = (size_t)B * N * H * DV;
        auto Q = load(dir + "/Q.f32", nqk), K = load(dir + "/K.f32", nqk), V = load(dir + "/V.f32", nvo);
        auto Qh = round_fp16(Q), Kh = round_fp16(K), Vh = round_fp16(V);
        auto orig = load(dir + "/O_original.f32", nvo);

        printf("%s  B=%d N=%d H=%d DK=%d DV=%d\n", dir.c_str(), B, N, H, DK, DV);
        for (int causal = 0; causal <= 1; ++causal) {
            const char *tag = causal ? "c" : "nc";
            auto O32 = attention(Q, K, V, B, N, H, DK, DV, causal);
            auto O16 = attention(Qh, Kh, Vh, B, N, H, DK, DV, causal);
            save(dir + "/O_" + tag + "_f32in.f32", O32);
            save(dir + "/O_" + tag + "_f16in.f32", O16);
            printf("  %-2s  |O_f32in - O_original| = %.3e   fp16 rounding cost |O_f16in - O_f32in| = %.3e\n",
                   tag, max_abs_diff(O32, orig), max_abs_diff(O16, O32));
        }
    }
    return 0;
}
