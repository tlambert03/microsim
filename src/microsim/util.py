from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt
import tqdm
from scipy import signal

from ._data_array import ArrayProtocol, DataArray

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray

    ShapeLike = Sequence[int]


def uniformly_spaced_coords(
    shape: tuple[int, ...] = (64, 128, 128),
    scale: tuple[float, ...] = (),
    extent: tuple[float, ...] = (),
    axes: str | Sequence[str] = "ZYX",
) -> dict[str, Sequence[float]]:
    # we now calculate the shape, scale, and extent based on input
    # where shape is the shape of the array, scale is the spacing between points
    # and extent is the total size of the array in each dimension (shape * scale)
    if not shape:
        if not extent:
            raise ValueError("Must provide either 'shape' or 'extent'")
        if not scale:
            raise ValueError("Must provide 'scale' along with 'extent'.")
        # scale = scale or ((1,) * len(extent))
        shape = tuple(int(x / s) for x, s in zip(extent, scale, strict=False))
    elif extent:
        if scale:
            warnings.warn(
                "Overdetermined: all three of shape, scale, and extent provided."
                "Ignoring value for extent.",
                stacklevel=2,
            )
        else:
            scale = tuple(x / s for x, s in zip(extent, shape, strict=False))
    elif not scale:
        scale = (1,) * len(shape)

    if not all(isinstance(i, int) for i in shape):
        raise TypeError(f"Shape must be a tuple of integers. Got {shape!r}")

    ndim = len(shape)
    if len(scale) != ndim:
        raise ValueError(f"length of scale and shape must match ({len(scale)}, {ndim})")
    if len(axes) < ndim:
        raise ValueError(f"Only {len(axes)} axes provided but got {ndim} dims")

    axes = axes[-ndim:]  # pick last ndim axes, in case there are too many provided.
    return {
        ax: np.arange(sh) * sc  # type: ignore
        for ax, sh, sc in zip(axes, shape, scale, strict=False)
    }


def uniformly_spaced_xarray(
    shape: tuple[int, ...] = (64, 128, 128),
    scale: tuple[float, ...] = (),
    extent: tuple[float, ...] = (),
    axes: str | Sequence[str] = "ZYX",
    array_creator: Callable[[ShapeLike], ArrayProtocol] = np.zeros,
) -> DataArray:
    coords = uniformly_spaced_coords(shape, scale, extent, axes)
    shape = tuple(len(c) for c in coords.values())
    return DataArray(array_creator(shape), coords=coords, attrs={"units": "um"})


def get_fftconvolve_shape(
    in1: npt.NDArray,
    in2: npt.NDArray,
    mode: Literal["full", "valid", "same"] = "full",
    axes: int | Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Get output shape of an fftconvolve operation (without performing it).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution. The default is over all axes.

    Returns
    -------
    tuple
        Tuple of ints, with output shape

    Raises
    ------
    ValueError
        If in1.shape and in2.shape are invalid for the provided mode.
    """
    if mode == "same":
        return in1.shape

    s1 = in1.shape
    s2 = in2.shape
    ndim = in1.ndim
    if axes is None:
        _axes = set(range(ndim))
    else:
        _axes = set(range(axes) if isinstance(axes, int) else axes)

    full_shape = tuple(
        max((s1[i], s2[i])) if i not in _axes else s1[i] + s2[i] - 1 for i in _axes
    )

    if mode == "valid":
        final_shape = tuple(
            full_shape[a] if a not in _axes else s1[a] - s2[a] + 1
            for a in range(len(full_shape))
        )
        if any(i <= 0 for i in final_shape):
            raise ValueError(
                "For 'valid' mode, one must be at least "
                "as large as the other in every dimension"
            )
    elif mode == "full":
        final_shape = full_shape

    else:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'")
    return final_shape


def _centered(arr: NDArray, newshape: ShapeLike) -> NDArray:
    """Return the center `newshape` portion of `arr`."""
    _newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind: NDArray = (currshape - _newshape) // 2
    endind: NDArray = startind + _newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _iter_block_locations(
    chunks: tuple[tuple[int, ...]],
) -> Iterator[tuple[list[tuple[int, int]], tuple[int, ...], ShapeLike]]:
    """Iterate block indices.

    Examples
    --------
    >>> chunks = ((2, 2), (3, 3), (4, 4, 4))
    >>> list(_iter_block_locations(chunks))
    [
        ([(0, 2), (0, 3), (0, 4)], (0, 0, 0), (2, 3, 4)),
        ([(0, 2), (0, 3), (4, 8)], (0, 0, 1), (2, 3, 4)),
        ([(0, 2), (0, 3), (8, 12)], (0, 0, 2), (2, 3, 4)),
        ...
    ]
    """
    starts = [(0, *tuple(np.cumsum(i))) for i in chunks]
    for block_id in itertools.product(*(range(len(c)) for c in chunks)):
        arr_slc = [(starts[ij][j], starts[ij][j + 1]) for ij, j in enumerate(block_id)]
        chunk_shape = tuple(chunks[ij][j] for ij, j in enumerate(block_id))
        yield arr_slc, block_id, chunk_shape


class Convolver(Protocol):
    def __call__(
        self, in1: NDArray, in2: NDArray, mode: Literal["full", "valid", "same"]
    ) -> NDArray: ...


def tiled_convolve(
    in1: npt.NDArray,
    in2: npt.NDArray,
    mode: Literal["full", "valid", "same"] = "full",
    chunks: tuple | None = None,
    func: Convolver = signal.convolve,
    dtype: DTypeLike | None = None,
) -> npt.NDArray:
    from dask.array.core import normalize_chunks

    if chunks is None:
        chunks = getattr(in1, "chunks", None) or (100,) * in1.ndim  # TODO: change 100

    _chunks: tuple[tuple[int, ...]] = normalize_chunks(chunks, in1.shape)

    final_shape = get_fftconvolve_shape(in1, in2, mode="full")

    out = np.zeros(final_shape, dtype=dtype)
    for loc, *_ in tqdm.tqdm(list(_iter_block_locations(_chunks))):
        block = np.asarray(in1[tuple(slice(*i) for i in loc)])
        result = func(block, in2, mode="full")
        if hasattr(result, "get"):
            result = result.get()
        out_idx = tuple(
            slice(i, i + s) for (i, _), s in zip(loc, result.shape, strict=False)
        )
        out[out_idx] += result
        del result

    if mode == "same":
        return _centered(out, in1.shape)
    elif mode == "valid":
        return _centered(out, get_fftconvolve_shape(in1, in2, mode="valid"))
    return out


# convenience function we'll use a couple times
def ortho_plot(img: ArrayProtocol, gamma: float = 0.5, mip: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    img = np.asarray(img)
    """Plot XY and XZ slices of a 3D array."""
    _, ax = plt.subplots(ncols=2, figsize=(10, 5))
    xy = img.max(axis=0) if mip else img[img.shape[0] // 2]
    xz = img.max(axis=1) if mip else img[:, img.shape[1] // 2]
    ax[0].imshow(xy, norm=PowerNorm(gamma))
    ax[1].imshow(xz, norm=PowerNorm(gamma))
    ax[0].set_title("XY slice")
    ax[1].set_title("XZ slice")
    plt.show()


ArrayType = TypeVar("ArrayType", bound=ArrayProtocol)


def downsample(
    array: ArrayType,
    factor: int | Sequence[int],
    method: Callable[
        [ArrayType, Sequence[int] | int | None, npt.DTypeLike], ArrayType
    ] = np.sum,
    dtype: npt.DTypeLike | None = None,
) -> ArrayType:
    binfactor = (factor,) * array.ndim if isinstance(factor, int) else factor
    new_shape = []
    for s, b in zip(array.shape, binfactor, strict=False):
        new_shape.extend([s // b, b])
    reshaped = cast("ArrayType", np.reshape(array, new_shape))
    for d in range(array.ndim):
        reshaped = method(reshaped, -1 * (d + 1), dtype)
    return reshaped
