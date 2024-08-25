from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import xarray
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from microsim.schema.backend import NumpyAPI

    def DataArray(
        data: Any,
        coords: Sequence[Sequence | xarray.DataArray] | Mapping | None = None,
        dims: str | Iterable[Hashable] | None = None,
        name: Hashable | None = None,
        attrs: Mapping | None = None,
    ) -> xarray.DataArray: ...

else:
    try:
        from .xarray_jax import DataArray
    except ImportError:
        from xarray import DataArray  # type: ignore[assignment]

__all__ = ["DataArray", "xrDataArray", "ArrayProtocol"]

xrDataArray = xarray.DataArray


@runtime_checkable
class DType(Protocol):
    itemsize: int
    name: str
    kind: str


@runtime_checkable
class ArrayProtocol(Protocol):
    """The basic protocol we need from an array-like object."""

    @property
    def dtype(self) -> np.dtype: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...

    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __array__(self) -> np.ndarray: ...
    def __mul__(self, other: Any) -> ArrayProtocol: ...


def from_cache(path: Path, xp: NumpyAPI | None = None) -> xrDataArray:
    data_set = xarray.open_zarr(path)
    # get the first data variable
    first_da = next(iter(data_set.data_vars))
    # xarray.open_zarr uses dask by default.  we may want that eventually
    # but for now, we force the computation
    da = data_set[first_da].compute()
    if xp is not None:
        da = DataArray(
            xp.asarray(da.data), coords=da.coords, dims=da.dims, attrs=da.attrs
        )
    da.attrs = _deserialize_attrs(da.attrs)
    return da  # type: ignore[no-any-return]


def to_cache(da: xrDataArray, path: Path, dtype: npt.DTypeLike | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    da = da.copy(deep=False)
    da.attrs = _serializable_attrs(da.attrs)
    if hasattr(da.data, "get"):
        da.data = da.data.get()
    da.to_zarr(path, mode="w")


def _serializable_attrs(attrs: Any) -> Any:
    """Make attrs serializable by json, including BaseModels."""
    if isinstance(attrs, dict):
        return {key: _serializable_attrs(value) for key, value in attrs.items()}
    elif isinstance(attrs, BaseModel):
        cls = type(attrs)
        data = attrs.model_dump(mode="json", exclude_unset=True)
        data["model_type"] = cls.__module__ + "." + cls.__name__
        return data
    elif isinstance(attrs, list | tuple):  # pragma: no cover
        return [_serializable_attrs(item) for item in attrs]
    else:
        return attrs  # pragma: no cover


def _deserialize_attrs(attrs: Any) -> Any:
    """Convert serialized attrs back to BaseModel instances."""
    if isinstance(attrs, dict):
        if model_type := attrs.pop("model_type", None):
            module, name = model_type.rsplit(".", 1)
            cls = getattr(__import__(module, fromlist=[name]), name)
            return cast("BaseModel", cls).model_validate(attrs)
        else:
            return {key: _deserialize_attrs(value) for key, value in attrs.items()}
    elif isinstance(attrs, list | tuple):  # pragma: no cover
        return type(attrs)(_deserialize_attrs(item) for item in attrs)
    else:
        return attrs  # pragma: no cover
