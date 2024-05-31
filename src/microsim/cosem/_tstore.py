from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import tensorstore as ts

from ._client import COSEM_BUCKET, COSEM_CACHE

if TYPE_CHECKING:
    from .models import CosemImage


DRIVERS = {
    "precomputed": "neuroglancer_precomputed",
    "n5": "n5",
    "zarr": "zarr",
}


def _kv_store(img: "CosemImage", level: int | None = None) -> dict:
    proto, bucket, path = img.url.partition(COSEM_BUCKET)
    path = path.lstrip("/")
    if level is not None and img.format != "precomputed":
        path += f"/s{level}"

    if (cached := COSEM_CACHE / path).exists():
        return {"driver": "file", "path": str(cached)}

    if not proto.startswith("s3"):
        raise ValueError(f"Unsupported protocol {proto!r}")
    return {
        "driver": "s3",
        "bucket": bucket,
        "path": path,
    }


def ts_spec(
    img: "CosemImage",
    level: int | None = None,
    cache_limit: float | None = None,
    **kwargs: Any,
) -> dict:
    if level is not None:
        try:
            lvl = img.scales[level]
            level = int(lvl.lstrip("bs"))
        except IndexError as e:
            raise IndexError(
                f"Level {level!r} not found in {img.name!r}. "
                f"Available levels are: {img.scales}"
            ) from e

    kvstore = _kv_store(img, level=level)
    if level is not None and img.format == "precomputed":
        if isinstance(level, str):
            level = level.lstrip("s")
        kwargs["scale_index"] = int(level)

    if cache_limit:
        kwargs["context"] = {"cache_pool": {"total_bytes_limit": cache_limit}}

    return {
        "driver": DRIVERS[img.format],
        "kvstore": kvstore,
        **kwargs,
    }


def read_tensorstore(
    img: "CosemImage",
    level: int | None = None,
    *,
    transpose: Sequence[str] | None = None,
    bin_mode: Literal["mode", "sum"] = "mode",
    cache_limit: float | None = 4e9,
) -> ts.TensorStore:
    data = ts.open(
        ts_spec(img, level=level, bin_mode=bin_mode, cache_limit=cache_limit)
    ).result()

    # "squeeze" the data (haven't found a tensorstore-native way to do this)
    # usually this is because of a single "channels" dim in precomputed formats.
    slices = tuple((slice(None) if s > 1 else 0) for s in data.shape)
    data = data[slices]

    # add dimension labels
    data = data[ts.d[:].label[tuple(img.grid_dims)]]

    # transpose the data if requested -- ("y", "x", "z") is a common transpose
    if transpose:
        data = data[ts.d[tuple(transpose)].transpose[:]]
    return data
