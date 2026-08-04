import torch


def make_case(N, d_k, d_v, boost_rows=None, factor=25.0, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(N, d_k)
    K = torch.randn(N, d_k)
    V = torch.randn(N, d_v)
    if boost_rows is not None:
        K[boost_rows] *= factor
    return Q, K, V

# name -> (Q, K, V, B_c, want_trace, causal)
CASES = {
    "baseline":         (*make_case(8, 4, 6), 3, True, False),
    "baseline_causal":  (*make_case(8, 4, 6), 3, True, True),
    "max_first":        (*make_case(8, 4, 6, slice(0, 3)), 3, True, False),
    "max_mid":          (*make_case(8, 4, 6, slice(3, 6)), 3, True, False),
    "max_last":         (*make_case(8, 4, 6, slice(6, 8)), 3, True, False),
    "blend":            (*make_case(8, 4, 6, slice(3, 6), factor=1.5), 3, True, False),
    "large":            (*make_case(512, 64, 64), 64, False, False),
    "large_causal":     (*make_case(512, 64, 64), 64, False, True),
    "large_odd":        (*make_case(517, 64, 48), 64, False, False),
    "massive":          (*make_case(4096, 64, 64), 64, False, False),
    "massive_causal":   (*make_case(4096, 64, 64), 64, False, True),
}