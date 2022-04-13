from __future__ import annotations

import warnings

# just rough notes for now
from typing import TYPE_CHECKING, Sequence, Tuple, Union

if TYPE_CHECKING:
    import xarray as xr

# sample = Sample()
# illum = Illumination()
# psf = PSF(
#     objective=Objective(),
#     coverslip=Coverslip(),
#     immersion_medium=ImmersionMedium(),
# )
# camera = Camera()
# optical_image = convolve(sample * illum, psf)
# digital_image = camera.simulate(optical_image)


# shape = (256, 512, 512)
# scale = (0.02, 0.01, 0.01)


def uniformly_spaced_array(
    shape: Tuple[int, ...] = (),
    scale: Tuple[float, ...] = (),
    extent: Tuple[int, ...] = (),
    axes: Union[str, Sequence[str]] = "ZYX",
) -> xr.DataArray:
    import dask.array as da
    import numpy as np
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
                "Overdetermined: all three of shape, extent, and scale provided."
                "Only using shape and scale."
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
    return xr.DataArray(da.zeros(shape), coords=coords)
