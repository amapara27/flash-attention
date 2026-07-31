import torch


def make_case(N, d_k, d_v, boost_rows=None, factor=25.0, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(N, d_k)
    K = torch.randn(N, d_k)
    V = torch.randn(N, d_v)
    if boost_rows is not None:
        K[boost_rows] *= factor
    return Q, K, V


# name -> (Q, K, V, B_c, trace?)
CASES = {
    "baseline":  (*make_case(8, 4, 6), 3, True),
    "max_first": (*make_case(8, 4, 6, slice(0, 3)), 3, True),
    "max_mid":   (*make_case(8, 4, 6, slice(3, 6)), 3, True),
    "max_last":  (*make_case(8, 4, 6, slice(6, 8)), 3, True),
    "blend":     (*make_case(8, 4, 6, slice(3, 6), factor=1.5), 3, True),
    "large":     (*make_case(512, 64, 64), 64, False),
    "large_ragged": (*make_case(517, 64, 48), 64, False),
}