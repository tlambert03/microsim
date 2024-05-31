import concurrent.futures
import logging
import shutil
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tensorstore as ts
import tqdm

from ._client import COSEM_BUCKET, COSEM_CACHE

if TYPE_CHECKING:
    from .models import CosemImage


DRIVERS = {
    "precomputed": "neuroglancer_precomputed",
    "n5": "n5",
    "zarr": "zarr",
}


def _kv_store(img: "CosemImage", level: int | None = None) -> dict:
    path = img.bucket_path.lstrip("/")
    if level is not None and img.format != "precomputed":
        path += f"/s{level}"

    if (cached := COSEM_CACHE / path).exists():
        return {"driver": "file", "path": str(cached)}

    return {
        "driver": "s3",
        "bucket": COSEM_BUCKET,
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
    # bin_mode: Literal["mode", "sum"] = "mode",
    cache_limit: float | None = 4e9,
) -> ts.TensorStore:
    """Read a COSEM image as a tensorstore.

    Parameters
    ----------
    img : CosemImage
        The image to read.
    level : int | None
        The scale level to read. If None, the highest resolution level is read.
    transpose : Sequence[str] | None
        The dimension order to transpose the data. If None, the default order is used.
    cache_limit : float | None
        The cache limit in bytes. If None, the default limit is used (4 GB).
    """
    data = ts.open(ts_spec(img, level=level, cache_limit=cache_limit)).result()

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


def new_like(
    store: ts.TensorStore,
    kvstore: str | dict | None = None,
    delete_existing: bool = True,
    dtype: Any = None,
) -> ts.TensorStore:
    """Create a new tensorstore like `store` with the data from `data`."""
    if not kvstore:
        kvstore = {"driver": "memory"}
    spec = {**store.spec().to_json(), "kvstore": kvstore, "dtype": dtype}
    spec["metadata"].pop("dataType", None)
    return ts.open(spec, create=True, delete_existing=delete_existing).result()


StartStop = tuple[int, int]
Indices = tuple[StartStop, ...]
IndicesPair = tuple[Indices, Indices]


def _chunk_idx_pairs(
    shape: Sequence[int],
    chunksize: Sequence[int],
    window: int | Sequence[int] = 2,
) -> list[IndicesPair]:
    """Return the index pairs for a given shape, chunksize, and window size.

    This function is used when downsampling a chunked array by binning blocks of data.

    Each item in the returned tuple is a pair of tuples, where the first tuple
    is the index of the source block and the second tuple is the index of the
    destination block.
    """
    if isinstance(window, int):
        window = (window,) * len(shape)

    index_pairs: list[IndicesPair] = []

    ranges = [
        range(0, dim, chunk) for dim, chunk in zip(shape, chunksize, strict=False)
    ]
    for indices in product(*ranges):
        input_slices = tuple(
            (idx, min(idx + chunk, dim))
            for idx, chunk, dim in zip(indices, chunksize, shape, strict=False)
        )
        output_slices = tuple(
            (idx // win, min(idx + chunk, dim) // win)
            for idx, chunk, dim, win in zip(
                indices, chunksize, shape, window, strict=False
            )
        )
        index_pairs.append((input_slices, output_slices))

    return index_pairs


def rebin(
    img: "CosemImage",
    *,
    max_level: int | None = None,
    dest: str | Path | None = None,
    skip_existing: bool = True,
    max_workers: int | None = None,
) -> None:
    dest = Path(dest) if dest else COSEM_CACHE
    arr_in = read_tensorstore(img, level=0).astype(ts.bool)
    for lvl in range(1, max_level or len(img.scales)):
        # determine chunk size of incoming array
        inchunk = cast(tuple, arr_in.chunk_layout.read_chunk.shape)
        # path of the destination array
        path = dest / (img.bucket_path.lstrip("/") + f"/sum{lvl}")
        kvstore = {"driver": "file", "path": str(path)}

        if path.exists() and skip_existing:
            arr_in = ts.open({"driver": "n5", "kvstore": kvstore}).result()
            logging.info(f"Skipping level {lvl} because it already exists.")
            continue

        # assuming all values are 1 or zero at lvl 0, then the maximum possible
        # value at lvl N is (2**ndim)**N.  So at lvl 3 the maximum value is 512 and
        # we jump to uint16.
        dtype = ts.uint8 if lvl < 3 else ts.uint16
        # to make our binned array as similar as possible to the one on S3, we create
        # a new array with (almost) the same spec as the original
        target = read_tensorstore(img, level=lvl)
        arr_out = new_like(target, kvstore=kvstore, dtype=dtype)

        # determine all of the input/output index pairs and process them concurrently
        items = [
            (*pair, arr_in, arr_out) for pair in _chunk_idx_pairs(arr_in.shape, inchunk)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                list(tqdm.tqdm(executor.map(_rebin_chunk, items), total=len(items)))
            except Exception:
                # clean up if something goes wrong
                shutil.rmtree(path)
                raise
        arr_in = arr_out


def _rebin_chunk(
    args: tuple[Indices, Indices, ts.TensorStore, ts.TensorStore],
) -> None:
    """Process a single chunk of data for rebinning.

    Reads a specific block from the input array, bins it, and writes it to the output
    array.
    """
    src, dest, in_arr, out_arr = args
    src_slice = tuple(slice(*x) for x in src)
    dest_slice = tuple(slice(*x) for x in dest)
    block = in_arr[src_slice]
    # TODO: see if tensorstore has a better way to check if a block is empty
    if np.any(block):  # type: ignore
        binned = block_bin(block.translate_to[0, 0, 0])
        if np.any(binned):
            # note, we could also use .write(), which is async...
            # but we're already in a thread, so it's not necessary?
            out_arr[dest_slice] = binned.astype(out_arr.dtype.numpy_dtype)  # type: ignore


def block_bin(
    block: ts.TensorStore, window: int | Sequence[int] = 2, ufunc: Callable = np.sum
) -> Any:
    """Apply a ufunc to a window of data in `block`.

    This function is used to downsize a block of data by applying a function to a
    window of data. The window size is determined by the `window` parameter.

    Note that the ufunc must accept an arraylike object and an axis parameter.
    """
    if isinstance(window, int):
        window = (window,) * len(block.shape)
    elif not len(window) == len(block.shape):
        raise ValueError("Window must have the same number of dimensions as the block.")

    new_shape = []
    for s, b in zip(block.shape, window, strict=True):
        new_shape.extend([s // b, b])

    # crop the last element if the shape is odd
    slc = tuple(slice(0, s * b) for s, b in zip(new_shape[::2], window, strict=True))
    sliced = block[slc]

    # reshape the block and sum the values
    reshaped_block = np.reshape(sliced, new_shape)  # type: ignore
    axes = tuple(range(1, reshaped_block.ndim, 2))
    return ufunc(reshaped_block, axis=axes)
