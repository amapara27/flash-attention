import shutil
from pathlib import Path

import numpy as np
import torch

from attention import attention
from flash_attention import flash_attention
from cases import CASES

OUT = Path("traces")
RAW = Path("../matrices")


def dump_all():
    # deletes files before creating new ones
    shutil.rmtree(RAW, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(exist_ok=True)

    for name, (Q, K, V, B_c, want_trace, causal) in CASES.items():
        if want_trace:
            assert Q.ndim == 2, f"{name}: tracing requires unbatched input"
            O, log = flash_attention(Q, K, V, B_c, causal=causal, trace=True)
            np.savez(
                OUT / f"{name}.npz",
                Q=Q.numpy(), K=K.numpy(), V=V.numpy(),
                O=O.numpy(), O_ref=attention(Q, K, V, causal=causal).numpy(),
                B_c=np.int32(B_c), scale=np.float32(Q.shape[-1] ** -0.5),
                causal=np.int32(causal),
                **{k: torch.stack([t[k] for t in log]).numpy()
                   for k in ("m", "ell", "O_acc", "corr", "rowsum")},
            )
            print(f"{name}: {len(log)} iters -> {OUT / f'{name}.npz'}")
        else:
            O = flash_attention(Q, K, V, B_c, causal=causal)
            print(f"{name}: no trace  {tuple(Q.shape)}")

        dump_raw(name, {"Q": Q, "K": K, "V": V, "O": O}, B_c, causal)


# permutes [B, H, N, d] -> [B, N, H, d] - head has to be batch dim for matmul braodcast, must swap here
def to_disk_layout(arr):
    return arr.permute(0, 2, 1, 3) if arr.ndim == 4 else arr


def dump_raw(name, arrays, B_c, causal):
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)

    disk = {k: to_disk_layout(v) for k, v in arrays.items()}

    for key, arr in disk.items():
        np.ascontiguousarray(arr.contiguous().numpy(), dtype=np.float32).tofile(
            d / f"{key}.f32")

    ndim = next(iter(disk.values())).ndim
    layout = "BNHd" if ndim == 4 else "Nd"

    with open(d / "meta.txt", "w") as f:
        # shapes are in DISK order, matching `layout`. variable length, so the
        # runner should read the token count rather than assume a rank.
        f.write(f"layout {layout}\n")
        for key, arr in disk.items():
            f.write(f"{key} {' '.join(map(str, arr.shape))}\n")
        f.write(f"B_c {B_c}\n")
        f.write(f"causal {int(causal)}\n")

    verify_raw(d, arrays, disk)


def verify_raw(d, arrays, disk):
    # declared shapes must be mutually consistent under one layout.
    # for BNHd every array shares B, N, H and may differ only in the last dim.
    shapes = [tuple(a.shape) for a in disk.values()]
    if len(shapes[0]) == 4:
        heads = {s[:3] for s in shapes}
        assert len(heads) == 1, \
            f"{d.name}: arrays disagree on (B, N, H): {dict(zip(disk, shapes))}"

    # bytes on disk must be in the declared element order.
    for key, orig in arrays.items():
        want = tuple(disk[key].shape)
        raw = np.fromfile(d / f"{key}.f32", dtype=np.float32)
        assert raw.size == int(np.prod(want)), \
            f"{d.name}/{key}: {raw.size} floats on disk, meta declares {want}"

        back = torch.from_numpy(raw.reshape(want))
        if len(want) == 4:
            back = back.permute(0, 2, 1, 3)   # BNHd -> BHNd, spelled out
        assert back.shape == orig.shape and torch.equal(back, orig), \
            f"{d.name}/{key}: on-disk layout does not match meta"


if __name__ == "__main__":
    dump_all()