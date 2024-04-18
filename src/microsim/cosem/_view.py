import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import dask
import xarray as xr

from ._dataset import CosemDataset

if TYPE_CHECKING:
    from datatree import DataTree


def load_view(
    dataset: str | CosemDataset,
    sources: Sequence[str] = (),
    # name: str | None = None,
    exclude: set[str] | None = None,
    extent: float | Sequence[float] | None = None,  # in nm around position, in XYZ
    position: Sequence[float] | None = None,  # in XYZ
    level: int = 0,
) -> xr.DataArray:
    if isinstance(dataset, str):
        ds = CosemDataset.fetch(dataset)
    else:
        ds = dataset

    # if name is not None:
    #     view = ds.view(name)
    #     position = view["position"]
    #     if not sources:
    #         sources = view["sources"]

    if exclude is None:
        exclude = set()

    # sources = [
    #     s.replace("fibsem-uint8", "fibsem-uint16")
    #     for s in sources
    #     if ds.sources.get(s, {}).get("contentType") not in exclude
    # ]
    images = ds.images

    _loaded: list[str] = []
    arrs: list[DataTree] = []
    for image in images:
        try:
            arrs.append(image.read(level))
            _loaded.append(image.name)
        except (NotImplementedError, KeyError):
            warnings.warn(f"Could not load image {image.name!r}", stacklevel=2)

    if not arrs:
        raise RuntimeError("No sources could be loaded")

    if len(arrs) > 1:
        with dask.config.set({"array.slicing.split_large_chunks": False}):
            stack = xr.concat(arrs, dim="source")
        stack.coords["source"] = _loaded
    else:
        stack = arrs[0]

    if extent is not None:
        if position is None:
            position = [stack.sizes[i] / 2 for i in "xyz"]
        stack = _crop_around(stack, position, extent)

    # .transpose("source", "y", "z", "x")
    return stack


def _crop_around(
    ary: xr.DataArray,
    position: Sequence[float],
    extent: float | Sequence[float],
    axes="xyz",
):
    """Crop dataarray around position."""
    if len(position) != 3:
        raise ValueError("position must be of length 3 (X, Y, Z)")
    if isinstance(extent, float | int):
        extent = (extent,) * 3
    if len(extent) != 3:
        raise ValueError("extent must be of length 3")

    slc = {
        ax: slice(p - e / 2, p + e / 2)
        for p, e, ax in zip(position, extent, axes, strict=False)
    }
    return ary.sel(**slc)
