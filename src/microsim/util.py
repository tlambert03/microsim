from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Callable, Iterator, Protocol, Sequence

import numpy as np
import numpy.typing as npt
from dask.array.core import normalize_chunks
from scipy import signal
from torch import le

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(a):
        return a


if TYPE_CHECKING:
    from typing import Literal

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    ShapeLike = Sequence[int]


def uniformly_spaced_coords(
    shape: tuple[int, ...] = (64, 128, 128),
    scale: tuple[float, ...] = (),
    extent: tuple[float, ...] = (),
    axes: str | Sequence[str] = "ZYX",
) -> list[tuple[str, npt.NDArray[np.int64]]]:
    # we now calculate the shape, scale, and extent based on input
    # where shape is the shape of the array, scale is the spacing between points
    # and extent is the total size of the array in each dimension (shape * scale)
    if not shape:
        if not extent:
            raise ValueError("Must provide either 'shape' or 'extent'")
        if not scale:
            raise ValueError("Must provide 'scale' along with 'extent'.")
        # scale = scale or ((1,) * len(extent))
        shape = tuple(int(x / s) for x, s in zip(extent, scale))
    elif extent:
        if scale:
            warnings.warn(
                "Overdetermined: all three of shape, scale, and extent provided."
                "Ignoring value for extent.",
                stacklevel=2,
            )
        else:
            scale = tuple(x / s for x, s in zip(extent, shape))
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
    return [(ax, np.arange(sh) * sc) for ax, sh, sc in zip(axes, shape, scale)]


def uniformly_spaced_xarray(
    shape: tuple[int, ...] = (64, 128, 128),
    scale: tuple[float, ...] = (),
    extent: tuple[float, ...] = (),
    axes: str | Sequence[str] = "ZYX",
    array_creator: Callable[[ShapeLike], ArrayLike] = np.zeros,
) -> xr.DataArray:
    import xarray as xr

    coords = uniformly_spaced_coords(shape, scale, extent, axes)
    shape = tuple(len(c) for _, c in coords)
    return xr.DataArray(array_creator(shape), coords=coords, attrs={"units": "um"})


def get_fftconvolve_shape(in1, in2, mode="full", axes=None):
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
    _axes = axes or range(ndim)

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
    in1: NDArray,
    in2: NDArray,
    mode: Literal["full", "valid", "same"] = "full",
    chunks: tuple | None = None,
    func: Convolver = signal.convolve,
    dtype: DTypeLike | None = None,
):
    if chunks is None:
        chunks = getattr(in1, "chunks", None) or (100,) * in1.ndim  # TODO: change 100

    _chunks: tuple[tuple[int, ...]] = normalize_chunks(chunks, in1.shape)

    final_shape = get_fftconvolve_shape(in1, in2, mode="full")

    out = np.zeros(final_shape, dtype=dtype)
    for loc, *_ in tqdm(list(_iter_block_locations(_chunks))):
        block = np.asarray(in1[tuple(slice(*i) for i in loc)])
        result = func(block, in2, mode="full")
        if hasattr(result, "get"):
            result = result.get()
        out_idx = tuple(slice(i, i + s) for (i, _), s in zip(loc, result.shape))
        out[out_idx] += result
        del result

    if mode == "same":
        return _centered(out, in1.shape)
    elif mode == "valid":
        return _centered(out, get_fftconvolve_shape(in1, in2, mode="valid"))
    return out


def make_confocal_psf(
    ex_wvl_um=0.475, em_wvl_um=0.525, pinhole_au=1.0, xp=np, **kwargs
):
    """Create a confocal PSF.

    This function creates a confocal PSF by multiplying the excitation PSF with
    the emission PSF convolved with a pinhole mask.

    All extra keyword arguments are passed to `vectorial_psf_centered`.
    """
    import tqdm
    from psfmodels import vectorial_psf_centered
    from scipy.signal import fftconvolve

    kwargs.pop("wvl", None)
    params: dict = kwargs.setdefault("params", {})
    na = params.setdefault("NA", 1.4)
    dxy = kwargs.setdefault("dxy", 0.01)

    print("making excitation PSF...")
    ex_psf = vectorial_psf_centered(wvl=ex_wvl_um, **kwargs)
    print("making emission PSF...")
    em_psf = vectorial_psf_centered(wvl=em_wvl_um, **kwargs)

    # The effective emission PSF is the regular emission PSF convolved with the
    # pinhole mask. The pinhole mask is a disk with diameter equal to the pinhole
    # size in AU, converted to pixels.
    pinhole = _pinhole_mask(
        nxy=ex_psf.shape[-1], pinhole_au=pinhole_au, wvl=em_wvl_um, na=na, dxy=dxy
    )
    pinhole = xp.asarray(pinhole)

    print("convolving em_psf with pinhole...")
    eff_em_psf = xp.empty_like(em_psf)
    for i in tqdm.trange(len(em_psf)):
        plane = fftconvolve(xp.asarray(em_psf[i]), pinhole, mode="same")
        eff_em_psf[i] = plane.get() if hasattr(plane, "get") else plane

    # The final PSF is the excitation PSF multiplied by the effective emission PSF.
    return ex_psf * eff_em_psf


def _pinhole_mask(
    nxy: int, pinhole_au: float, wvl: float, na: float, dxy: float, xp=np
):
    """Create a 2D circular pinhole mask of specified `pinhole_au`."""
    pinhole_size = pinhole_au * 0.61 * wvl / na
    pinhole_px = pinhole_size / dxy

    x = xp.arange(nxy) - nxy // 2
    xx, yy = xp.meshgrid(x, x)
    r = xp.sqrt(xx**2 + yy**2)
    return (r <= pinhole_px).astype(int)


# convenience function we'll use a couple times
def ortho_plot(img, gamma: float = 0.5, mip: bool = False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    """Plot XY and XZ slices of a 3D array."""
    _, ax = plt.subplots(ncols=2, figsize=(10, 5))
    xy = img.max(axis=0) if mip else img[img.shape[0] // 2]
    xz = img.max(axis=1) if mip else img[:, img.shape[1] // 2]
    ax[0].imshow(xy, norm=PowerNorm(gamma))
    ax[1].imshow(xz, norm=PowerNorm(gamma))
    ax[0].set_title("XY slice")
    ax[1].set_title("XZ slice")
    plt.show()


def downsample(
    array: np.ndarray, factor: int | Sequence[int], method=np.sum, dtype=None
) -> np.ndarray:
    binfactor = (factor,) * array.ndim if isinstance(factor, int) else factor
    new_shape = []
    for s, b in zip(array.shape, binfactor):
        new_shape.extend([s // b, b])
    reshaped = np.reshape(array, new_shape)
    for d in range(array.ndim):
        reshaped = method(reshaped, axis=-1 * (d + 1), dtype=dtype)
    return reshaped
