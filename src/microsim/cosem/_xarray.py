"""Code mostly borrowed/adapted from fibsem_tools.

...to minimize dependencies and avoid pydantic hard pinning.

https://github.com/janelia-cellmap/fibsem-tools

Copyright 2019 Howard Hughes Medical Institute
MIT License

"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TCH003
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from datatree import DataTree
from pydantic import BaseModel

if TYPE_CHECKING:
    PathLike = Path | str


def access_zarr(
    store: PathLike, path: PathLike = "", **kwargs: Any
) -> zarr.Array | zarr.Group:
    if isinstance(store, Path | str):
        store = str(store)
        if ".n5" in store:
            store, path = str(store).split("n5")
            store += "n5"
            store = zarr.N5FSStore(store, anon=True)
        else:
            raise NotImplementedError(f"Can only read n5 sources, not {store!r}")

    # set default dimension separator to /
    if "shape" in kwargs and "dimension_separator" not in kwargs:
        kwargs["dimension_separator"] = "/"

    attrs = kwargs.pop("attrs", {})
    access_mode = kwargs.pop("mode", "a")

    array_or_group = zarr.open(store, path=str(path), **kwargs, mode=access_mode)

    if access_mode != "r" and len(attrs) > 0:
        array_or_group.attrs.update(attrs)
    return array_or_group


def stt_coord(
    length: int, dim: str, scale: float, translate: float, units: str
) -> xr.DataArray:
    return xr.DataArray(
        (np.arange(length) * scale) + translate, dims=(dim,), attrs={"units": units}
    )


class STTransform(BaseModel):
    order: Literal["C", "F"] | None = "C"
    axes: Sequence[str]
    units: Sequence[str]
    translate: Sequence[float]
    scale: Sequence[float]

    def to_coords(self, shape: tuple[int, ...]) -> list[xr.DataArray]:
        axes = self.axes if self.order == "C" else reversed(self.axes)
        return [
            stt_coord(
                shape[idx],
                dim=k,
                scale=self.scale[idx],
                translate=self.translate[idx],
                units=self.units[idx],
            )
            for idx, k in enumerate(axes)
        ]


# def access_parent(node: zarr.Array | zarr.Group, **kwargs: Any) -> zarr.Group:
#     """Get the parent (zarr.Group) of a Zarr array or group."""
#     parent_path = "/".join(node.path.split("/")[:-1])
#     return access_zarr(store=node.store, path=parent_path, **kwargs)


def infer_coords(array: zarr.Array) -> list[xr.DataArray]:
    # group = access_parent(array, mode="r")

    if (transform := array.attrs.get("transform", None)) is not None:
        return STTransform.model_validate(transform).to_coords(array.shape)

    raise NotImplementedError('No "transform" attribute found in array.')


def create_datatree(
    element: zarr.Group,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataTree:
    if name is None:
        name = element.basename

    nodes: dict[str, xr.Dataset | xr.DataArray | DataTree | None] = {
        name: create_dataarray(
            array,
            chunks=chunks,
            use_dask=use_dask,
            attrs=None,
            name="data",
        )
        for name, array in element.arrays()
    }
    if attrs is None:
        attrs = dict(element.attrs)

    # insert root element
    nodes["/"] = xr.Dataset(attrs=attrs)
    dtree = DataTree.from_dict(nodes, name=name)
    return dtree


def create_dataarray(
    element: zarr.Array | zarr.Group | da.Array,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> xr.DataArray:
    name = name or element.basename
    coords = infer_coords(element)
    if attrs is None:
        attrs = dict(getattr(element, "attrs", {}))
    if use_dask:
        element = da.from_array(element, chunks=chunks, inline_array=True)
    result = xr.DataArray(element, coords=coords, attrs=attrs, name=name)
    return result


def zarr_to_xarray(
    element: zarr.Array | zarr.Group,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> xr.DataArray | DataTree:
    if isinstance(element, zarr.Group):
        return create_datatree(
            element,
            chunks=chunks,
            attrs=attrs,
            use_dask=use_dask,
            name=name,
        )
    if isinstance(element, zarr.Array):
        return create_dataarray(
            element,
            chunks=chunks,
            attrs=attrs,
            use_dask=use_dask,
            name=name,
        )
    raise ValueError(
        "This function only accepts instances of zarr.Group and zarr.Array. ",
        f"Got {type(element)} instead.",
    )


def read_xarray(
    path: PathLike,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> xr.DataArray | DataTree:
    element = access_zarr(path)
    return zarr_to_xarray(
        element,
        chunks=chunks,
        use_dask=use_dask,
        attrs=attrs,
        name=name,
    )


if __name__ == "__main__":
    from rich import print

    g = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/chrom_pred"
    print(read_xarray(g))
