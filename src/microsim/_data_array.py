from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import xarray

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    import numpy as np

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
