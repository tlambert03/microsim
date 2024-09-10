from __future__ import annotations

import itertools
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from urllib import parse, request
from urllib.error import HTTPError

import numpy as np
import numpy.typing as npt
import platformdirs
import tqdm
from scipy import signal

from ._data_array import ArrayProtocol, DataArray, xrDataArray

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray

    ShapeLike = Sequence[int]


# don't use this directly... it's patched during tests
# use cache_path() instead
_MICROSIM_CACHE = platformdirs.user_cache_path("microsim")


def microsim_cache(subdir: Literal["psf", "ground_truth"] | None = None) -> Path:
    """Return the microsim cache path.

    If `subdir` is provided, return the path to the specified subdirectory.
    (We use literal here to ensure that only the specified values are allowed.)
    """
    if subdir:
        return _MICROSIM_CACHE / subdir
    return _MICROSIM_CACHE


def clear_cache(pattern: str | None = None) -> None:
    """Clear the microsim cache."""
    if pattern:
        for p in microsim_cache().glob(pattern):
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p, ignore_errors=True)
    else:
        shutil.rmtree(microsim_cache(), ignore_errors=True)


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
    attrs: Mapping | None = None,
) -> xrDataArray:
    coords = uniformly_spaced_coords(shape, scale, extent, axes)
    shape = tuple(len(c) for c in coords.values())
    return DataArray(array_creator(shape), dims=tuple(axes), coords=coords, attrs=attrs)


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
def ortho_plot(
    img: ArrayProtocol,
    gamma: float = 1,
    mip: bool = False,
    cmap: str | list[str] | None = None,
    *,
    title: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (8, 8),
    z: int | None = None,
) -> None:
    """Plot XY and XZ slices of a 3D array."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if isinstance(img, xrDataArray):
        img = img.data
    if hasattr(img, "get"):
        img = img.get()
    img = np.asarray(img).squeeze()
    cmap = [cmap] if isinstance(cmap, str) else cmap
    if img.ndim == 3:
        channels = [img]
        cm_list = cmap if cmap is not None else ["white"]
    elif img.ndim == 4:
        channels = list(img)
        colors = ["green", "magenta", "cyan", "yellow", "red", "blue"]
        cm_list = cmap if cmap is not None else colors
    else:
        raise ValueError("Input must be a 3D or 4D array")

    # Initialize RGB images for xy and xz
    nz, ny, nx = channels[0].shape
    midz = nz // 2 if z is None else z
    midy, midx = ny // 2, nx // 2
    xy_rgb = np.zeros((ny, nx, 3))
    xz_rgb = np.zeros((nz, nx, 3))
    yz_rgb = np.zeros((nz, ny, 3))

    for img, cmap in zip(channels, cm_list, strict=False):
        xy = np.max(img, axis=0) if mip else img[midz]
        xz = np.max(img, axis=1) if mip else img[:, midy]
        yz = np.max(img, axis=2) if mip else img[:, :, midx]

        # Normalize the images to the range [0, 1]
        mi, ma = np.percentile(xy, (0.1, 99.9))
        xy = (xy - mi) / (ma - mi)
        xz = (xz - mi) / (ma - mi)
        yz = (yz - mi) / (ma - mi)

        # Apply gamma correction
        xy = np.power(xy, 1 / gamma)
        xz = np.power(xz, 1 / gamma)
        yz = np.power(yz, 1 / gamma)

        # Convert the grayscale images to RGB using the specified colormap
        cm = LinearSegmentedColormap.from_list("_cmap", ["black", cmap])
        xy_rgb += cm(xy)[..., :3]  # Exclude alpha channel
        xz_rgb += cm(xz)[..., :3]  # Exclude alpha channel
        yz_rgb += cm(yz)[..., :3]  # Exclude alpha channel

    # Clip the values to the range [0, 1]
    xy_rgb = np.clip(xy_rgb, 0, 1)
    xz_rgb = np.clip(xz_rgb, 0, 1)
    yz_rgb = np.clip(yz_rgb, 0, 1)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(nx, nz),
        height_ratios=(ny, nz),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.01,
        hspace=0.01,
    )
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_yz = fig.add_subplot(gs[0, 1], sharey=ax_xy)
    ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_xy)

    ax_xy.imshow(xy_rgb, interpolation="none")
    ax_xy.set_title("XY")
    ax_xy.get_xaxis().set_visible(False)

    # ax_yz.imshow(np.rot90(yz_rgb), interpolation="none")
    ax_yz.imshow(np.flipud(np.rot90(yz_rgb)), interpolation="none")
    ax_yz.set_title("YZ")
    ax_yz.get_yaxis().set_visible(False)

    ax_xz.imshow(xz_rgb, interpolation="none")
    ax_xz.set_title("XZ", y=0, loc="left", color="gray")

    if not mip:
        # Assuming 'mid_x' is the index where the YZ slice is taken
        ax_xy.axvline(x=midx, color="yellow", linestyle="--", linewidth=1, alpha=0.3)
        ax_xy.axhline(y=midy, color="yellow", linestyle="--", linewidth=1, alpha=0.3)
        ax_xz.axvline(x=midx, color="yellow", linestyle="--", linewidth=1, alpha=0.3)
        ax_xz.axhline(y=midz, color="yellow", linestyle="--", linewidth=1, alpha=0.3)
        ax_yz.axvline(x=midz, color="yellow", linestyle="--", linewidth=1, alpha=0.3)
        ax_yz.axhline(y=midy, color="yellow", linestyle="--", linewidth=1, alpha=0.3)

    # Remove spines to make the plot tighter
    for ax in [ax_xy, ax_yz, ax_xz]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16)
    if show:
        plt.show()


def ndview(ary: Any, cmap: Any | None = None) -> None:
    """View any array using ndv.imshow.

    This function is a thin wrapper around `ndv.imshow`.
    """
    try:
        import ndv
        import qtpy
        import vispy.app
    except ImportError as e:
        raise ImportError(
            "Please `pip install 'ndv[pyqt,vispy]' to use this function."
        ) from e

    vispy.use(qtpy.API_NAME)
    ndv.imshow(ary, cmap=cmap)


ArrayType = TypeVar("ArrayType", bound=ArrayProtocol)


def bin_window(
    array: npt.NDArray,
    window: int | Sequence[int],
    dtype: npt.DTypeLike | None = None,
    method: str | Callable = "sum",
) -> npt.NDArray:
    """Bin an nd-array by applying `method` over `window`."""
    # TODO: deal with xarray

    binwindow = (window,) * array.ndim if isinstance(window, int) else window
    new_shape = []
    for s, b in zip(array.shape, binwindow, strict=False):
        new_shape.extend([s // b, b])

    sliced = array[
        tuple(slice(0, s * b) for s, b in zip(new_shape[::2], binwindow, strict=True))
    ]
    reshaped = np.reshape(sliced, new_shape)

    if callable(method):
        f = method
    elif method == "mode":
        # round and cast to int before calling bincount
        reshaped = np.round(reshaped).astype(np.int32, casting="unsafe")

        def f(a: npt.NDArray, axis: int) -> npt.NDArray:
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis, a)
    else:
        f = getattr(np, method)
    axes = tuple(range(1, reshaped.ndim, 2))
    result = np.apply_over_axes(f, reshaped, axes).squeeze()
    if dtype is not None:
        result = result.astype(dtype)
    return result


def norm_name(name: str) -> str:
    """Normalize a name to something easily searchable."""
    name = str(name).lower()
    for char in " -/\\()[],;:!?@#$%^&*+=|<>'\"":
        name = name.replace(char, "_")
    return name


def http_get(url: str, params: dict | None = None) -> bytes:
    """API like requests.get but with standard-library urllib."""
    if params:
        url += "?" + parse.urlencode(params)

    with request.urlopen(url) as response:
        if not 200 <= response.getcode() < 300:  # pragma: no cover
            raise HTTPError(
                url, response.getcode(), "HTTP request failed", response.headers, None
            )
        return cast(bytes, response.read())
