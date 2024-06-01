import json
import shutil
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, TypeAdapter

ZarrWriteModes = Literal["w", "w-", "a", "a-", "r+", "r"]

if TYPE_CHECKING:
    import xarray as xr


class ArrayProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> np.dtype: ...
    def __array__(self) -> np.ndarray: ...
    def __mul__(self, other: Any) -> "ArrayProtocol": ...


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
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def sizes(self) -> MappingProxyType[str, int]:
        return MappingProxyType({k: len(v) for k, v in self.coords.items()})

    def __add__(self, other: Any) -> "DataArray":
        return DataArray(self.data + other, self.coords, self.attrs)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        data = self.data.get() if hasattr(self.data, "get") else self.data
        return np.asanyarray(data, dtype=dtype)

    def to_tiff(
        self, path: str | PathLike[str], description: str | None = None
    ) -> None:
        import tifffile as tf

        data = np.asanyarray(self)
        try:
            # sometimes description fails
            tf.imwrite(path, data, description=description)
        except ValueError:
            tf.imwrite(path, data)

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
    ) -> "xr.DataArray":
        import xarray as xr

        attrs = {**self.attrs, **(attrs or {})}
        return xr.DataArray(np.asanyarray(self), coords=self.coords, attrs=attrs)

    @classmethod
    def from_xarray(cls, ary: "xr.DataArray") -> "DataArray":
        coords = {dim: np.asarray(ary.coords[dim]) for dim in ary.dims}
        return cls(ary.data, coords, ary.attrs)

    @classmethod
    def from_cache(self, path: Path) -> "DataArray":
        import tensorstore as ts

        store = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(path)},
            },
        ).result()
        data = store.read().result()
        meta = json.loads(store.kvstore.read(".zattrs").result().value)
        coords = {k: np.asarray(v) for k, v in meta["coords"].items()}
        attrs = meta["attrs"]
        if "space" in attrs:
            from microsim.schema.space import Space

            attrs["space"] = TypeAdapter(Space).validate_python(attrs["space"])
        return DataArray(data, coords=coords, attrs=attrs)

    def to_cache(self, path: Path, dtype: npt.DTypeLike | None = None) -> None:
        import tensorstore as ts

        path.mkdir(parents=True, exist_ok=True)
        dtype = np.dtype(dtype or self.dtype)
        store = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(path)},
                "metadata": {
                    "dtype": dtype.str,
                    "shape": self.shape,
                    "chunks": (256,) * len(self.shape),
                    "dimension_separator": "/",
                },
                "create": True,
            },
        ).result()

        try:
            zattrs = self._serializeable_metadata()
            store.kvstore.write(".zattrs", json.dumps(zattrs)).result()
            store[:] = np.asarray(self, dtype=dtype)
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise

    def _serializeable_metadata(self) -> dict[str, Any]:
        attrs = {}
        for k, v in self.attrs.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, BaseModel):
                v = v.model_dump(mode="json")
            attrs[k] = v
        return {
            "coords": {k: list(v) for k, v in self.coords.items()},
            "attrs": attrs,
        }
