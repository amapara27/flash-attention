import shutil
from pathlib import Path

import numpy as np
import torch

from attention import attention
from flash_attention import flash_attention
from cases import CASES, FP32_REF

OUT = Path("traces")
RAW = Path("../matrices")


def dump_all():
    # Ghost .f32 files from renamed or removed cases silently pass/fail against
    # stale data. Adding batch dims changes every raw file's shape, so this is
    # exactly the moment that bites. Nuke and rebuild.
    #
    # NOTE: this deletes all of RAW. If anything else lives in ../matrices,
    # move it or narrow this to the case dirs.
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

        # calculate fp32 O matrix from true fp32 references - determine what fp16 calculations cost in kernel
        if name.startswith("fp16_"):
            Q_true, K_true, V_true = FP32_REF[name]
            O_disk = attention(Q_true, K_true, V_true, causal=causal)
        else:
            O_disk = O

        dump_raw(name, {"Q": Q, "K": K, "V": V, "O": O_disk}, B_c, causal)


def to_disk_layout(arr):
    # [B, H, N, d] -> [B, N, H, d]. 2D passes through.
    # converts to correct format - head must have been batch dim for reference calcs
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