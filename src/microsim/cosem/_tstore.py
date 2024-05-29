from typing import TYPE_CHECKING, Any

import tensorstore as ts

if TYPE_CHECKING:
    from ._dataset import CosemImage


DRIVERS = {
    "precomputed": "neuroglancer_precomputed",
    "n5": "n5",
    "zarr": "zarr",
}


def _kv_store(img: "CosemImage") -> dict:
    proto, bucket, path = img.url.partition("janelia-cosem-datasets")
    if not proto.startswith("s3"):
        raise ValueError(f"Unsupported protocol {proto!r}")
    return {
        "driver": "s3",
        "bucket": bucket,
        "path": path.lstrip("/"),
    }


def ts_spec(img: "CosemImage", level: int | str | None = None, **kwargs: Any) -> dict:
    kvstore = _kv_store(img)
    if level is not None:
        if img.format == "precomputed":
            if isinstance(level, str):
                level = level.lstrip("s")
            kwargs["scale_index"] = int(level)
        else:
            if isinstance(level, int):
                level = f"s{level}"
            kvstore["path"] = f"{kvstore['path']}/{level}"

    return {
        "driver": DRIVERS[img.format],
        "kvstore": kvstore,
        "context": {"cache_pool": {"total_bytes_limit": 2**32}},  # 4GB
        **kwargs,
    }


def read_tensorstore(
    img: "CosemImage", level: int | str | None = None
) -> ts.TensorStore:
    data = ts.open(ts_spec(img, level=level)).result()
    # "squeeze" the data
    slices = tuple((slice(None) if s > 1 else 0) for s in data.shape)
    data = data[slices]
    return data[ts.d[:].label[tuple(img.grid_dims)]]
