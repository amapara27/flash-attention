from pathlib import Path
import numpy as np
import torch
from attention import attention
from flash_attention import flash_attention
from cases import CASES

OUT = Path("traces")
RAW = Path("../matrices")

def dump_all():
    OUT.mkdir(exist_ok=True)
    for name, (Q, K, V, B_c, want_trace, causal) in CASES.items():
        if want_trace:
            O, log = flash_attention(Q, K, V, B_c, causal=causal, trace=True)
            np.savez(
                OUT / f"{name}.npz",
                Q=Q.numpy(), K=K.numpy(), V=V.numpy(),
                O=O.numpy(), O_ref=attention(Q, K, V, causal=causal).numpy(),
                B_c=np.int32(B_c), scale=np.float32(Q.shape[1] ** -0.5),
                causal=np.int32(causal),
                **{k: torch.stack([t[k] for t in log]).numpy()
                   for k in ("m", "ell", "O_acc", "corr", "rowsum")},
            )
            print(f"{name}: {len(log)} iters -> {OUT / f'{name}.npz'}")
        else:
            O = flash_attention(Q, K, V, B_c, causal=causal)
            print(f"{name}: no trace")

        dump_raw(name, {"Q": Q, "K": K, "V": V, "O": O})

def dump_raw(name, arrays):
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        np.ascontiguousarray(arr, dtype=np.float32).tofile(d / f"{key}.f32")
    with open(d / "meta.txt", "w") as f:
        for key, arr in arrays.items():
            f.write(f"{key} {' '.join(map(str, arr.shape))}\n")


if __name__ == "__main__":
    dump_all()