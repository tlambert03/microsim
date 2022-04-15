from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import ArrayLike

    ShapeLike = Sequence[int]


def uniformly_spaced_xarray(
    shape: Tuple[int, ...] = (64, 128, 128),
    scale: Tuple[float, ...] = (),
    extent: Tuple[int, ...] = (),
    axes: Union[str, Sequence[str]] = "ZYX",
    array_creator: Callable[[ShapeLike], ArrayLike] = np.zeros,
) -> xr.DataArray:
    import xarray as xr

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
                "Ignoring value for extent."
            )
        else:
            scale = tuple(x / s for x, s in zip(extent, shape))
    elif not scale:
        scale = (1,) * len(shape)

    if not all(isinstance(i, int) for i in shape):
        raise TypeError(f"Shape must be a tuple of integers. Got {shape!r}")

    ndim = len(shape)
    assert (
        len(scale) == ndim
    ), f"length of scale and shape must match ({len(scale)}, {ndim})"
    assert len(axes) >= ndim, f"Only {len(axes)} axes provided but got {ndim} dims"

    axes = axes[-ndim:]  # pick last ndim axes, in case there are too many provided.
    coords = [(ax, np.arange(sh) * sc) for ax, sh, sc in zip(axes, shape, scale)]
    return xr.DataArray(array_creator(shape), coords=coords, attrs={"units": "um"})
