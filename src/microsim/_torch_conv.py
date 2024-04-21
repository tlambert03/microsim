from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType

    from microsim._data_array import ArrayProtocol


def torch_conv(
    in1: ArrayProtocol,
    in2: ArrayProtocol,
    mode: str = "same",
    device: DeviceLikeType | None = None,
) -> torch.Tensor:
    if in1.ndim != in2.ndim:
        raise ValueError("a and k should have the same dimensionality")
    if in1.ndim not in [1, 2, 3]:
        raise ValueError("a and k should be 2D or 3D")
    if mode not in ["same", "full", "valid"]:
        raise ValueError("mode must be one of ['same', 'full', 'valid']")
    if mode == "full":
        raise NotImplementedError("full mode not implemented yet")

    sig = torch.tensor(in1, device=device) if not isinstance(in1, torch.Tensor) else in1
    k = torch.tensor(in2, device=device) if not isinstance(in2, torch.Tensor) else in2

    fa = torch.fft.rfftn(sig.float())
    fb = torch.fft.rfftn(k.float())
    fc = fa * fb
    result = torch.fft.irfftn(fc, s=sig.shape)
    return result


if __name__ == "__main__":
    import numpy as np

    ndim = 3
    side = 3
    sig = np.arange(side**ndim).reshape(*([side] * ndim))
    kern = np.zeros_like(sig)
    middle = (side // 2,) * ndim
    kern[middle] = 1

    print(sig)
    print(kern)
    r = torch_conv(sig, kern)
    print(sig.shape, r.shape)
    print(r)
