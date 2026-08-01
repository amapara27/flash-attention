import torch
import torch.nn.functional as F

from attention import attention
from flash_attention import flash_attention
from cases import CASES


def check_all():
    for name, (Q, K, V, B_c, _) in CASES.items():
        O = flash_attention(Q, K, V, B_c)
        err = (O - attention(Q, K, V)).abs().max().item()
        assert err < 1e-5, f"{name}: {err:.2e}"

        print(f"{name:14s} N={Q.shape[0]:4d} B_c={B_c:3d} err={err:.2e}")
        print(torch.allclose(O, F.scaled_dot_product_attention(Q, K, V), atol=1e-7))


if __name__ == "__main__":
    check_all()