from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pyvkfft.fft as vkfft

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(in1.ndim)
    ]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False, tune=tune)

    return _apply_conv_mode(ret, s1, s2, mode, axes).copy()


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
    sorted_axes: bool = False,
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

    if sorted_axes:
        axes.sort()

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

    # fshape = list(shape)
    if calc_fast_len:
        raise NotImplementedError("calc_fast_len not implemented yet")
        # Speed up FFT by padding to optimal size.
        # fshape = [sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]

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


def _apply_conv_mode(
    ret: ArrayProtocol,
    s1: tuple[int, ...],
    s2: tuple[int, ...],
    mode: Mode,
    axes: list[int],
) -> ArrayProtocol:
    """Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
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
    else:
        raise ValueError("acceptable mode flags are 'valid'," " 'same', or 'full'")


def _centered(arr: ArrayProtocol, newshape: Sequence[int]) -> ArrayProtocol:
    # Return the center newshape portion of the array.
    newshape_ = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape_) // 2
    endind = startind + newshape_
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]  # type: ignore
