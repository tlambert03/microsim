from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, no_type_check
from unittest.mock import patch

from scipy.fft import next_fast_len
from scipy.signal import fftconvolve
from scipy.signal._signaltools import _centered

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    import numpy.typing as npt

    from microsim._data_array import ArrayProtocol
    from microsim.schema.backend import NumpyAPI

# LRU cache of Fourier-transformed kernels (see `_cached_kernel_ft`)
_KERNEL_FT_CACHE: OrderedDict[Hashable, ArrayProtocol] = OrderedDict()


@no_type_check
def _apply_conv_mode_no_copy(ret, s1, s2, mode, axes, xp=None):
    # same as scipy.signal._signaltools._apply_conv_mode but without the copy.
    # `xp` is accepted (and ignored) for compatibility with scipy >=1.15, which
    # added an array-namespace argument and copies via `xp_copy(..., xp=xp)`.
    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
            for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid)
    raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")


def patched_fftconvolve(
    in1: Any, in2: Any, mode: str = "full", axes: int | npt.ArrayLike | None = None
) -> Any:
    # scipy.signal.fftconvolve assumes that all arraylike objects have a .copy() method
    # which is not the case for some backends (like Torch).
    # This function patches the _apply_conv_mode function to avoid calling .copy()
    # after cropping the result to the desired shape.
    with patch("scipy.signal._signaltools._apply_conv_mode", _apply_conv_mode_no_copy):
        return fftconvolve(in1, in2, mode=mode, axes=axes)


def fftconvolve_same_pairs(
    images: Sequence[ArrayProtocol],
    kernels: Mapping[tuple[int, int], tuple[Hashable, ArrayProtocol]],
    n_out: int,
    xp: NumpyAPI,
    cache_mb: float = 0,
) -> list[list[ArrayProtocol]]:
    """Return `out[o][i] = fftconvolve(images[i], kernel_oi, mode="same")`, faster.

    Used to convolve each fluorophore's ground truth (`images`) with its PSF in each
    of `n_out` channels. Pair-by-pair scipy calls would re-transform each image for
    every output, and each kernel on every call. Instead, all pairs share one
    (minimal) FFT shape, each image is transformed once, and kernel transforms are
    cached by `key` (up to `cache_mb`) across calls. `kernels` maps `(o, i)` to
    `(key, kernel_oi)`, where `key` uniquely identifies the kernel; missing pairs
    give zeros.
    """
    if not images:
        return [[] for _ in range(n_out)]
    shape = images[0].shape
    fshape = _same_conv_fshape(shape, (k.shape for _, k in kernels.values()))
    out: list[list] = [[None] * len(images) for _ in range(n_out)]
    # image-outer, so only one image transform is held in memory at a time
    for i, image in enumerate(images):
        image_ft = None
        for o in range(n_out):
            if (o, i) not in kernels:
                out[o][i] = xp.zeros_like(image)
                continue
            key, kernel = kernels[(o, i)]
            kernel = xp.asarray(kernel, dtype=xp.result_type(image, kernel))
            if image_ft is None:
                image_ft = xp.fft.rfftn(image, s=fshape)
            kernel_ft = _cached_kernel_ft(key, kernel, fshape, xp, cache_mb)
            full = xp.fft.irfftn(image_ft * kernel_ft, s=fshape)
            out[o][i] = _crop_same(full, shape, kernel.shape)
    return out


def _same_conv_fshape(
    shape: Sequence[int], kernel_shapes: Iterable[Sequence[int]]
) -> tuple[int, ...]:
    """Smallest fast FFT shape for which circular == linear `mode="same"` conv."""
    # The full linear convolution has length s1 + s2 - 1 along each axis, of which
    # `mode="same"` keeps s1 samples starting at (s2 - 1) // 2.  With an FFT of
    # length n, samples beyond n wrap around to the start, which is harmless as long
    # as they land before the kept window: n >= s1 + s2 - 1 - (s2 - 1) // 2.
    # This is ~25% smaller per axis than the n >= s1 + s2 - 1 that scipy uses.
    need = [0] * len(shape)
    for kshape in kernel_shapes:
        for ax, (s1, s2) in enumerate(zip(shape, kshape, strict=True)):
            need[ax] = max(need[ax], s1 + s2 - 1 - (s2 - 1) // 2)
    return tuple(next_fast_len(n, real=True) for n in need)


def _crop_same(full: Any, shape: Sequence[int], kernel_shape: Sequence[int]) -> Any:
    """Crop a circular convolution result to the `mode="same"` window."""
    starts = [(s2 - 1) // 2 for s2 in kernel_shape]
    return full[tuple(slice(a, a + s) for a, s in zip(starts, shape, strict=True))]


def _cached_kernel_ft(
    key: Hashable,
    kernel: ArrayProtocol,
    fshape: tuple[int, ...],
    xp: NumpyAPI,
    cache_mb: float,
) -> ArrayProtocol:
    """Return `rfftn(kernel, s=fshape)`, cached by `key` (identifying the kernel)."""
    full_key = (key, fshape, str(kernel.dtype), type(xp))
    if (kernel_ft := _KERNEL_FT_CACHE.get(full_key)) is not None:
        _KERNEL_FT_CACHE.move_to_end(full_key)
        return kernel_ft

    new_ft: ArrayProtocol = xp.fft.rfftn(kernel, s=fshape)
    max_bytes = cache_mb * 1e6
    if _nbytes(new_ft) <= max_bytes:
        _KERNEL_FT_CACHE[full_key] = new_ft
        while sum(_nbytes(v) for v in _KERNEL_FT_CACHE.values()) > max_bytes:
            _KERNEL_FT_CACHE.popitem(last=False)
    return new_ft


def _nbytes(ary: ArrayProtocol) -> int:
    return ary.size * ary.dtype.itemsize
