from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from attr import has

ZarrWriteModes = Literal["w", "w-", "a", "a-", "r+", "r"]

if TYPE_CHECKING:
    import xarray as xr


class ArrayProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    def __array__(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class DataArray:
    """Data array with coordinates and attributes.

    This is the minimal xarray.DataArray API that we use
    unfortunately, xarray casts data to numpy arrays, making it hard to use as a
    container for jax, cupy, etc.
    https://github.com/google/jax/issues/17107
    https://github.com/pydata/xarray/issues/7848
    """

    data: ArrayProtocol
    coords: Mapping[str, Sequence[float]] = field(default_factory=dict)
    attrs: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.coords)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def sizes(self) -> MappingProxyType[str, int]:
        return MappingProxyType({k: len(v) for k, v in self.coords.items()})

    def __add__(self, other: Any) -> "DataArray":
        return DataArray(self.data + other, self.coords, self.attrs)

    def __array__(self) -> Any:
        return np.asanyarray(self.data)

    def to_tiff(
        self, path: str | PathLike[str], description: str | None = None
    ) -> None:
        import tifffile as tf

        tf.imwrite(path, self._to_cpu(), description=description)

    def to_zarr(
        self,
        store: str | PathLike[str],
        mode: ZarrWriteModes | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.to_xarray(attrs=attrs).to_zarr(store, mode=mode)

    def to_netcdf(
        self,
        path: str | PathLike[str],
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.to_xarray(attrs=attrs).to_netcdf(path)

    def to_xarray(
        self,
        attrs: Mapping[str, Any] | None = None,
        get: bool = True,
    ) -> "xr.DataArray":
        import xarray as xr

        attrs = {**self.attrs, **(attrs or {})}
        data = self._to_cpu() if get else self.data
        return xr.DataArray(data, coords=self.coords, attrs=attrs)

    def _to_cpu(self) -> np.ndarray:
        data = self.data
        if callable(gettr := getattr(data, "get", None)):
            # for GPU arrays
            data = gettr()
        return data
