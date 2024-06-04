from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import xarray

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
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...

    # strides: tuple[int, ...]
    # size: int
    # itemsize: int
    # nbytes: int

    # def __len__(self) -> int: ...
    # def __getitem__(self, key: Any) -> Any: ...
    # def __setitem__(self, key: Any, value: Any) -> None: ...
    def __array__(self) -> np.ndarray: ...
    def __mul__(self, other: Any) -> Self: ...
