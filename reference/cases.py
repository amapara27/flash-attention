import torch


def make_case(N, d_k, d_v, boost_rows=None, factor=25.0, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(N, d_k)
    K = torch.randn(N, d_k)
    V = torch.randn(N, d_v)
    if boost_rows is not None:
        K[boost_rows] *= factor
    return Q, K, V


def make_batched_case(B, H, N, d_k, d_v, boost_rows=None, factor=25.0, seed=0):
    Qs, Ks, Vs = [], [], []

    for b in range(B):
        for h in range(H):
            lin = b * H + h
            q, k, v = make_case(N, d_k, d_v, seed=seed + lin)

            if boost_rows is not None:
                rows = torch.arange(N)[boost_rows]
                rows = (rows + lin) % N   # shift is 0 for (0, 0)
                k[rows] *= factor

            Qs.append(q)
            Ks.append(k)
            Vs.append(v)

    def pack(mats, d):
        return torch.stack(mats).reshape(B, H, N, d).contiguous()

    return pack(Qs, d_k), pack(Ks, d_k), pack(Vs, d_v)


# rounds fp32 vals into fp16 vals
def to_fp16_storage(*mats):
    return tuple(m.half().float() for m in mats)


# name -> (Q, K, V, B_c, want_trace, causal)
CASES = {
    # ---- traced: line-by-line debug target ----
    "baseline":            (*make_case(8, 4, 6), 3, True, False),
    "baseline_causal":     (*make_case(8, 4, 6), 3, True, True),
    "max_mid":             (*make_case(8, 4, 6, slice(3, 6)), 3, True, False),

    # ---- unbatched: ragged tile + d_k != d_v, and the perf shape ----
    "large_odd":           (*make_case(517, 64, 48), 64, False, False),
    "massive_causal":      (*make_case(4096, 64, 64), 64, False, True),

    # ---- batched: ragged tile + d_k!=d_v + per-head boost, all in one ----
    "batch_odd":           (*make_batched_case(2, 3, 517, 64, 48, slice(3, 6)), 64, False, False),

    # ---- batched: the realistic perf/profiling shape ----
    "batch_massive_causal": (*make_batched_case(1, 4, 4096, 64, 64), 64, False, True),

    # ---- fp16 storage precision: same shapes as the two perf cases above ----
    "fp16_massive_causal":       (*to_fp16_storage(*make_case(4096, 64, 64)), 64, False, True),
    "fp16_batch_massive_causal": (*to_fp16_storage(*make_batched_case(1, 4, 4096, 64, 64)), 64, False, True),
}

# fp32 references for fp16 vals
FP32_REF = {
    "fp16_massive_causal":       make_case(4096, 64, 64),
    "fp16_batch_massive_causal": make_batched_case(1, 4, 4096, 64, 64),
}