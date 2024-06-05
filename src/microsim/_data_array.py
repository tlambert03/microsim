from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import xarray

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

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


def from_cache(path: Path) -> xrDataArray:
    data_set = xarray.open_zarr(path)
    # get the first data variable
    first_da = next(iter(data_set.data_vars))
    # xarray.open_zarr uses dask by default.  we may want that eventually
    # but for now, we force the computation
    return data_set[first_da].compute()  # type: ignore[no-any-return]


def to_cache(da: xrDataArray, path: Path, dtype: npt.DTypeLike | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    da.to_zarr(path, mode="w")
