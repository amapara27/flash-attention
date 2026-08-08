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


# name -> (Q, K, V, B_c, want_trace, causal)
CASES = {
    # unbatched originals
    "baseline":            (*make_case(8, 4, 6), 3, True, False),
    "baseline_causal":     (*make_case(8, 4, 6), 3, True, True),
    "max_first":           (*make_case(8, 4, 6, slice(0, 3)), 3, True, False),
    "max_mid":             (*make_case(8, 4, 6, slice(3, 6)), 3, True, False),
    "max_last":            (*make_case(8, 4, 6, slice(6, 8)), 3, True, False),
    "blend":               (*make_case(8, 4, 6, slice(3, 6), factor=1.5), 3, True, False),
    "large":               (*make_case(512, 64, 64), 64, False, False),
    "large_causal":        (*make_case(512, 64, 64), 64, False, True),
    "large_odd":           (*make_case(517, 64, 48), 64, False, False),
    "massive":             (*make_case(4096, 64, 64), 64, False, False),
    "massive_causal":      (*make_case(4096, 64, 64), 64, False, True),

    # batched small
    "batch_small":         (*make_batched_case(2, 3, 8, 4, 6), 3, False, False),
    "batch_small_causal":  (*make_batched_case(2, 3, 8, 4, 6), 3, False, True),

    # boost rows shift per head -- the head-isolation probe
    "batch_boost":         (*make_batched_case(2, 3, 8, 4, 6, slice(3, 6)), 3, False, False),

    # d_k != d_v batched: catches reusing the Q/K offset for V or O
    "batch_odd":           (*make_batched_case(2, 3, 517, 64, 48), 64, False, False),

    # batched - realistic head counts
    "batch_heads":         (*make_batched_case(1, 32, 512, 64, 64), 64, False, False),
    "batch_heads_causal":  (*make_batched_case(1, 32, 512, 64, 64), 64, False, True),
    "batch_multi":         (*make_batched_case(2, 8, 512, 64, 64), 64, False, True),

    # perfect shape
    "batch_massive_causal": (*make_batched_case(1, 4, 4096, 64, 64), 64, False, True),
}