"""Module slated for removal. see ._dataset instead."""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

from fibsem_tools import read_xarray
from typing_extensions import TypedDict

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import xarray as xr

with open(Path(__file__).parent / "organelles.json") as fh:
    ORGANELLES = json.load(fh)

ORGANELLE_KEY = {x["file_name"]: x["full_name"] for x in ORGANELLES}
GH_API = "https://raw.githubusercontent.com/janelia-cosem/fibsem-metadata/stable/api"
COSEM_S3 = "s3://janelia-cosem-datasets"


class DatasetMetadata(TypedDict):
    title: str
    id: str
    imaging: dict
    sample: dict
    institution: list[str]
    softwareAvailability: str
    DOI: list[dict]
    publications: list[dict]


class DatasetView(TypedDict):
    """A view of a cosem dataset.

    - sources: suggested layers
    - position: [X, Y, Z] centerpoint of the feature
    - scale: nm/pixel at which to show the view
    - orientation: always seems to be [1, 0, 0, 0].
    """

    name: str
    description: str
    sources: list[str]
    position: list[float] | None
    scale: float | None
    orientation: list[float]


class Source(TypedDict):
    name: str
    description: str
    url: str
    format: str
    transform: dict
    sampleType: str
    contentType: str
    displaySettings: dict
    subsources: list


class DatasetManifest(TypedDict):
    name: str
    metadata: DatasetMetadata
    sources: dict[str, Source]
    views: list[DatasetView]


class CosemDataset:
    def __init__(self, id: str) -> None:
        self.id = id

    def read_xarray(self):
        read_xarray()

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def title(self) -> str:
        return self.metadata["title"]

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return (
            f"<CosemDataset '{self}' sources: {len(self.sources)}, "
            f"views: {len(self.views)}>"
        )

    @property
    def manifest(self) -> DatasetManifest:
        return get_manifest(self.id)

    @property
    def metadata(self) -> DatasetMetadata:
        return self.manifest["metadata"]

    @property
    def thumbnail(self) -> np.ndarray:
        return get_thumbnail(self.id)

    @cached_property
    def views(self) -> list[DatasetView]:
        return self.manifest["views"]

    @property
    def sources(self) -> dict[str, Source]:
        return self.manifest["sources"]

    def read_source(self, key: str, level=0) -> xr.DataArray:
        source = self.sources[key]
        if source["format"] != "n5":
            raise NotImplementedError(
                f"Can only read n5 sources, (not {source['format']!r})"
            )

        return read_xarray(f"{source['url']}/s{level}", storage_options={"anon": True})

    def view(self, name: str) -> DatasetView:
        for d in self.views:
            if d["name"].lower().startswith(name.lower()):
                return d
        raise KeyError(f"No view named/starting with {name!r}")

    def load_view(
        self,
        name: str | None = None,
        sources: Sequence[str] = (),
        position: Sequence[float] | None = None,
        exclude: set[str] | None = None,
        extent=1000,  # in nm around position
        level=0,
    ):
        return load_view(
            self,
            name=name,
            sources=sources,
            position=position,
            exclude=exclude,
            extent=extent,
            level=level,
        )


@lru_cache
def get_datasets() -> dict[str, str]:
    """Retrieve available datasets from janelia-cosem/fibsem-metadata."""
    with urlopen(f"{GH_API}/index.json") as r:
        return json.load(r).get("datasets")


@lru_cache(maxsize=64)
def get_manifest(dataset: str) -> DatasetManifest:
    """Get manifest for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset ID, e.g. "jrc_hela-3".

    Returns
    -------
    Dict[str, str]
        Useful keys include:

        * views: a curated list of views with:

    """
    with urlopen(f"{GH_API}/{dataset}/manifest.json") as r:
        return json.load(r)


@lru_cache(maxsize=64)
def get_thumbnail(dataset: str) -> np.ndarray:
    import imageio

    return imageio.imread(f"{GH_API}/{dataset}/thumbnail.jpg")


def load_view(
    dataset: str | CosemDataset,
    sources: Sequence[str] = (),
    name: str | None = None,
    exclude: set[str] | None = None,
    extent: float | Sequence[float] | None = None,  # in nm around position, in XYZ
    position: Sequence[float] | None = None,  # in XYZ
    level=0,
):
    import dask
    import xarray as xr

    ds = CosemDataset(dataset) if isinstance(dataset, str) else dataset
    if name is not None:
        view = ds.view(name)
        position = view["position"]
        if not sources:
            sources = view["sources"]

    if exclude is None:
        exclude = set()

    sources = [
        s.replace("fibsem-uint8", "fibsem-uint16")
        for s in sources
        if ds.sources.get(s, {}).get("contentType") not in exclude
    ]
    _loaded: list[str] = []
    arrs: list[xr.DataArray] = []
    for source in sources:
        try:
            arrs.append(ds.read_source(source, level))
            _loaded.append(source)
        except (NotImplementedError, KeyError):
            warnings.warn(f"Could not load source {source!r}", stacklevel=2)

    if not arrs:
        raise RuntimeError("No sources could be loaded")

    if len(arrs) > 1:
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            stack = xr.concat(arrs, dim="source")
        stack.coords["source"] = _loaded
    else:
        stack = arrs[0]

    if extent is not None:
        if position is None:
            position = [stack.sizes[i] / 2 for i in "xyz"]
        stack = _crop_around(stack, position, extent)

    # .transpose("source", "y", "z", "x")
    return stack


def _crop_around(
    ary: xr.DataArray,
    position: Sequence[float],
    extent: float | Sequence[float],
    axes="xyz",
):
    """Crop dataarray around position."""
    if len(position) != 3:
        raise ValueError("position must be of length 3 (X, Y, Z)")
    if isinstance(extent, float | int):
        extent = (extent,) * 3
    if len(extent) != 3:
        raise ValueError("extent must be of length 3")

    slc = {
        ax: slice(p - e / 2, p + e / 2)
        for p, e, ax in zip(position, extent, axes, strict=False)
    }
    return ary.sel(**slc)


def read_dataset(dataset: str, source: str, level: int = 0) -> xr.Dataset:
    uri = f"{COSEM_S3}/{dataset}/{dataset}.n5/{source}/s{level}"
    return read_xarray(uri, storage_options={"anon": True})


SAMPLES = {
    "hela_cytosol": {
        "dataset": "jrc_hela-3",
        "position": [35026, 1533, 18200],
        "extent": [17361, 2296, 16082],
        # "extent": [4361, 1296, 4082],
        # "sources": ["er_seg", "mt-out_seg"],
    }
}


jrc_hela_2_roi = {
    "dataset": "jrc_hela-2",
    "slices": (slice(300, 1000), slice(20, 220), slice(2100, 2800)),
    "sources": ["mito-mem_seg", "mt-out_seg"],
}


def _load_local(dataset, sources, slices=None, b=4):
    import dask.array as da
    import zarr.convenience

    _dir = Path(__file__).parent.parent.parent
    zas = [
        zarr.convenience.open_array(_dir / f"{dataset}" / f"{source}_b{b}.zarr")
        for source in sources
    ]
    za = da.stack(zas)
    if slices := (slice(None), *slices):
        za = za[slices]
    return za.transpose(0, 2, 1, 3)
