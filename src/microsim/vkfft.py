from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pyvkfft.fft as vkfft

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyopencl.array as cla

    from ._data_array import ArrayProtocol

    Mode = Literal["full", "valid", "same"]


def fftconvolve(
    in1: ArrayProtocol,
    in2: ArrayProtocol,
    mode: Literal["full", "valid", "same"] = "full",
    axes: int | Sequence[int] | None = None,
    tune: bool = False,
) -> ArrayProtocol:
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    if in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    if mode not in ["same", "full", "valid"]:
        raise ValueError("mode must be one of ['same', 'full', 'valid']")

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes)

    s1 = in1.shape
    s2 = in2.shape

    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(in1.ndim)
    ]
    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True, tune=tune)
    slc = _get_slice(ret, s1, s2, mode, axes)
    cropped = ret if slc is None else ret[slc]
    return cropped.map_to_host()


def _init_nd_axes(in1: ArrayProtocol, axes: int | Sequence[int] | None) -> list[int]:
    if axes is None:
        return list(range(in1.ndim))

    axes = (axes,) if isinstance(axes, int) else axes
    axes = [a + in1.ndim if a < 0 else a for a in axes]

    if any(a >= in1.ndim or a < 0 for a in axes):
        raise ValueError("axes exceeds dimensionality of input")
    if len(set(axes)) != len(axes):
        raise ValueError("all axes must be unique")
    return list(axes)


def _init_freq_conv_axes(
    in1: ArrayProtocol,
    in2: ArrayProtocol,
    mode: Mode,
    axes: int | Sequence[int] | None,
) -> tuple[ArrayProtocol, ArrayProtocol, list[int]]:
    s1 = in1.shape
    s2 = in2.shape

    noaxes = axes is None
    axes = _init_nd_axes(in1, axes=axes)
    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")

    # Axes of length 1 can rely on broadcasting rules for multiply,
    # no fft needed.
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

    if not all(
        s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
        for a in range(in1.ndim)
        if a not in axes
    ):
        raise ValueError("incompatible shapes for in1 and in2:" f" {s1} and {s2}")

    if mode == "valid":
        raise NotImplementedError("mode 'valid' not implemented yet")
        # Check that input sizes are compatible with 'valid' mode.
        # if _inputs_swap_needed(mode, s1, s2, axes=axes):
        #     in1, in2 = in2, in1

    return in1, in2, axes


def _pad_with_zeros(src: ArrayProtocol, sh: Sequence[int]) -> ArrayProtocol:
    if sh == src.shape:
        return src

    if vkfft.has_pycuda:
        if isinstance(src, vkfft.cua.GPUArray):
            raise NotImplementedError("pad_with_zeros not implemented for GPUArray")
    if vkfft.has_opencl:
        if isinstance(src, vkfft.cla.Array):
            return pad(src, sh, value=0)
    if vkfft.has_cupy:
        if isinstance(src, vkfft.cp.ndarray):
            raise NotImplementedError("pad_with_zeros not implemented for cupy")


def _freq_domain_conv(
    in1: ArrayProtocol,
    in2: ArrayProtocol,
    axes: Sequence[int],
    shape: Sequence[int],
    calc_fast_len: bool = False,
    tune: bool = False,
) -> ArrayProtocol:
    """Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.
    tune : bool, optional
        If `True`, tune the FFTs for optimal performance.
        Default is `False`.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.
    """
    if not len(axes):
        return in1 * in2

    complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [_next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    in1 = _pad_with_zeros(in1, fshape)
    in2 = _pad_with_zeros(in2, fshape)

    if not complex_result:
        if len(axes) != in1.ndim:
            raise NotImplementedError("sub-axes with rfft not implemented yet")
        sp1 = vkfft.rfftn(in1, tune=tune)
        sp2 = vkfft.rfftn(in2, tune=tune)
        ret = vkfft.irfftn(sp1 * sp2, tune=tune)
    else:
        sp1 = vkfft.fftn(in1, axes=axes, tune=tune)
        sp2 = vkfft.fftn(in2, axes=axes, tune=tune)
        ret = vkfft.ifftn(sp1 * sp2, axes=axes, tune=tune)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret  # type: ignore


def _next_fast_len(target, real=False):
    """Find the next fast size of input data to ``fft``, for zero-padding, etc."""
    if target <= 2:
        return target

    while True:
        n = target
        while n % 2 == 0:
            n //= 2
        while n % 3 == 0:
            n //= 3
        while n % 5 == 0:
            n //= 5
        if n == 1:
            return target
        target += 1


def _get_slice(
    ret: ArrayProtocol,
    s1: tuple[int, ...],
    s2: tuple[int, ...],
    mode: Mode,
    axes: list[int],
) -> tuple | None:
    if mode == "full":
        return None
    elif mode == "same":
        return _center_slice(ret, s1)
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
            for a in range(ret.ndim)
        ]
        return _center_slice(ret, shape_valid)
    else:
        raise ValueError("acceptable mode flags are 'valid'," " 'same', or 'full'")


def _center_slice(arr: ArrayProtocol, newshape: Sequence[int]) -> tuple:
    # Return the center newshape portion of the array.
    newshape_ = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape_) // 2
    endind = startind + newshape_
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    print(arr.shape, myslice)
    return tuple(myslice)


def pad(
    src: cla.Array,
    final_shape=tuple[int, ...],
    value=0,
    queue=None,
    block=False,
) -> cla.Array:
    """Pad an array to a new shape with a constant value.

    For now, only OpenCL arrays are supported.
    The image is padded at the end of each axis with the given value.
    """
    import pyopencl.array as cla

    assert isinstance(src, cla.Array)
    if queue is None:
        queue = src.queue

    dst = cla.zeros(queue, final_shape, src.dtype) + value
    last = src.shape[-3] if src.ndim > 2 else 1
    region = (src.dtype.itemsize * src.shape[-1], src.shape[-2], last)
    return _copy_array(src, dst, (0, 0, 0), (0, 0, 0), region, block=block)


def _copy_array(
    src: cla.Array,
    dst: cla.Array,
    src_origin: tuple,
    dst_origin: tuple,
    region: tuple,
    block=False,
) -> cla.Array:
    import pyopencl as cl

    n_bytes = src.dtype.itemsize
    ev = cl.enqueue_copy(
        src.queue,
        dst.data,
        src.data,
        src_origin=src_origin,
        dst_origin=dst_origin,
        region=region,
        src_pitches=(n_bytes * src.shape[-1], n_bytes * src.shape[-1] * src.shape[-2]),
        dst_pitches=(n_bytes * dst.shape[-1], n_bytes * dst.shape[-1] * dst.shape[-2]),
    )
    if block:
        ev.wait()
    return dst
