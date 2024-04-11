from itertools import product
from os import makedirs, path

import numpy as np
import zarr.convenience
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from microsim.samples import _cosem
from microsim.schema.detectors._simulate import bin

datasets = ["jrc_hela-2"]
sources = [
    # "mt-out_seg",
    # "mito-mem_seg",
    # "er-mem_seg",
    # "np_seg",
    # "pm_seg",
    # "ne-mem_seg",
    "chrom_seg",
    "ribo_seg",
]


def convert(data: tuple[str, str], target: str = ".") -> None:
    print(data, "started")
    dataset, source = data
    makedirs(path.join(target, dataset), exist_ok=True)
    vol = _cosem.CosemDataset(dataset).read_source(source)
    print("reading...")
    with ProgressBar():
        _vol = vol.data.astype(bool).compute()
    print("converted")
    binned = bin(_vol, 2, dtype=np.uint8)
    print("binned")
    zarr.convenience.save_array(
        f"{dataset}/{source}_b2.zarr", binned, chunks=(128, 128, 128)
    )
    print("saved")
    binned = bin(binned, 2, dtype=np.uint8)
    print("binned again")
    zarr.convenience.save_array(
        f"{dataset}/{source}_b4.zarr", binned, chunks=(64, 64, 64)
    )
    print(data, "DONE")


if __name__ == "__main__":
    client = Client(threads_per_worker=4, n_workers=2)
    print(client)

    for d in product(datasets, sources):
        try:
            convert(d)
        except Exception as e:
            print(f"ERROR {e}")
    print("all done")
