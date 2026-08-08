import torch
import torch.nn.functional as F

from attention import attention
from flash_attention import flash_attention
from cases import CASES


def sdpa_ref(Q, K, V, causal):
    pad = 4 - Q.ndim
    q, k, v = (t[(None,) * pad] for t in (Q, K, V))
    O = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return O.reshape(Q.shape[:-1] + (V.shape[-1],))


# max error per (b, h) slize
def per_slice_err(O, O_ref):
    err = (O - O_ref).abs()
    return err.reshape(-1, *err.shape[-2:]).amax(dim=(-2, -1))


def check_all():
    for name, (Q, K, V, B_c, want_trace, causal) in CASES.items():
        O = flash_attention(Q, K, V, B_c, causal=causal)
        O_ref = attention(Q, K, V, causal=causal)

        err = (O - O_ref).abs().max().item()
        assert err < 1e-4, f"{name}: vs attention {err:.2e}"

        sdpa_err = (O - sdpa_ref(Q, K, V, causal)).abs().max().item()
        assert sdpa_err < 1e-4, f"{name}: vs sdpa {sdpa_err:.2e}"

        tag = f"{name:22s} {str(tuple(Q.shape)):22s} B_c={B_c:3d} causal={int(causal)}"

        if Q.ndim == 2:
            print(f"{tag} err={err:.2e}")
            continue

        O_00 = flash_attention(Q[0, 0], K[0, 0], V[0, 0], B_c, causal=causal)
        slice_drift = (O_00 - O[0, 0]).abs().max().item()
        assert slice_drift < 1e-5, \
            f"{name}: batched path diverges from 2D path on (0,0): {slice_drift:.2e}"

        per = per_slice_err(O, O_ref)
        H = Q.shape[1]
        worst = int(per.argmax())
        print(f"{tag} err={err:.2e} sdpa={sdpa_err:.2e} "
              f"worst=(b{worst // H},h{worst % H}) drift={slice_drift:.1e} "
              f"spread=[{per.min():.2e}, {per.max():.2e}]")


# makes sure first slice is accurate - baseline for following comparisons
def check_slice_00_matches_unbatched():
    from cases import make_case, make_batched_case

    for kwargs in ({}, {"boost_rows": slice(3, 6)}):
        a = make_case(8, 4, 6, **kwargs)
        b = make_batched_case(2, 3, 8, 4, 6, **kwargs)
        for t_a, t_b, nm in zip(a, b, "QKV"):
            assert torch.equal(t_a, t_b[0, 0]), f"{nm} slice (0,0) drifted {kwargs}"
    print("slice (0,0) construction matches make_case bit-exactly")


# ensures all (b, h) slices hold distinct data
def check_heads_differ():
    from cases import make_batched_case

    Q, K, V = make_batched_case(2, 3, 8, 4, 6, boost_rows=slice(3, 6))
    flat = K.reshape(-1, 8, 4)
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            assert not torch.equal(flat[i], flat[j]), f"K slices {i},{j} identical"
    # and the boost landed in a different row set per head
    peak = K.abs().amax(-1).argmax(-1).reshape(-1)
    assert len(set(peak.tolist())) > 1, "boost rows did not shift across heads"
    print(f"all slices distinct; boost peak row per head: {peak.tolist()}")


if __name__ == "__main__":
    check_slice_00_matches_unbatched()
    check_heads_differ()
    check_all()