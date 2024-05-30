import datetime
import json
import logging
import re
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import botocore.response
import tensorstore as ts
import tqdm
from pydantic import BaseModel, computed_field

if TYPE_CHECKING:
    import botocore.client
    import botocore.response
    import numpy as np
    import supabase
    from boto3.resources.base import ServiceResource
    from datatree import DataTree


@cache
def _supabase() -> "supabase.Client":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        from supabase import Client
    except ImportError as e:
        raise ImportError(
            "To use cosem data, please `pip install microsim[cosem]`"
        ) from e

    with urllib.request.urlopen(
        "https://openorganelle.janelia.org/static/js/4743.a9f85e14.chunk.js"
    ) as response:
        if response.status != 200:
            raise ValueError("Failed to fetch Supabase URL and key")
        text = response.read().decode("utf-8")
    key = text.split("SUPABASE_KEY:")[1].split(",")[0].strip("\"'")
    url = text.split("SUPABASE_URL:")[1].split(",")[0].strip("\"'")
    return Client(url, key)


@cache
def _s3_client() -> "botocore.client.BaseClient":
    logging.getLogger("botocore").setLevel(logging.WARNING)
    import boto3
    from botocore import UNSIGNED, client

    return boto3.client("s3", config=client.Config(signature_version=UNSIGNED))


@cache
def _bucket(name="janelia-cosem-datasets") -> "ServiceResource":
    logging.getLogger("botocore").setLevel(logging.WARNING)
    import boto3

    return boto3.resource("s3").Bucket(name)


def fetch_s3(url: str) -> "botocore.response.StreamingBody":
    proto, _, bucket, key = url.split("/", 3)
    if not proto.startswith("s3"):
        raise ValueError(f"Unsupported protocol {proto!r}")

    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    response_meta = obj["ResponseMetadata"]
    if not (status := response_meta.get("HTTPStatusCode")) == 200:
        raise ValueError(f"Failed to fetch {url!r} with status {status}")
    return obj["Body"]


def fetch_s3_json(url: str) -> Any:
    return json.load(fetch_s3(url))


DATASETS_QUERY = """
name,
description,
thumbnail_url,
sample:sample(
    name,
    description,
    protocol,
    type,
    subtype,
    organism
),
image_acquisition:image_acquisition(
    name,
    start_date,
    grid_axes,
    grid_spacing,
    grid_dimensions,
    grid_spacing_unit,
    grid_dimensions_unit
),
images:image(
    name,
    description,
    url,
    format,
    grid_scale,
    grid_translation,
    grid_dims,
    grid_units,
    sample_type,
    content_type
)
""".strip().replace("\n", "")

VIEWS_QUERY = """
name,
dataset_name,
description,
thumbnail_url,
position,
scale,
orientation,
taxa:taxon(name, short_name),
created_at,
images:image(
    name,
    description,
    url,
    format,
    grid_scale,
    grid_translation,
    grid_dims,
    grid_units,
    sample_type,
    content_type
)
""".strip().replace("\n", "")


@cache
def fetch_datasets() -> Mapping[str, "CosemDataset"]:
    """Fetch all dataset metadata from the COSEM database."""
    response = _supabase().from_("dataset").select(DATASETS_QUERY).execute()
    datasets: dict[str, CosemDataset] = {}
    for x in response.data:
        ds = CosemDataset.model_validate(x)
        datasets[ds.name] = ds
    return MappingProxyType(datasets)


@cache
def fetch_views() -> list["CosemView"]:
    """Fetch all view metadata from the COSEM database."""
    response = _supabase().from_("view").select(VIEWS_QUERY).execute()
    return [CosemView.model_validate(x) for x in response.data]


@cache
def organelles() -> Mapping[str, str]:
    orgs: defaultdict[str, list[CosemView]] = defaultdict(list)
    for view in fetch_views():
        for taxon in view.taxa:
            orgs[taxon.name].append(view)
    return MappingProxyType(orgs)


# ------------------------ MODELS ------------------------


class CosemSample(BaseModel):
    name: str
    description: str
    protocol: str
    type: list[str] | None
    subtype: list[str] | None
    organism: list[str]


class CosemImageAcquisition(BaseModel):
    name: str
    start_date: datetime.datetime
    grid_axes: list[str]
    grid_spacing: list[float]
    grid_dimensions: list[float]
    grid_spacing_unit: str
    grid_dimensions_unit: str


import hashlib


def calculate_etag(file_path: str | Path) -> str:
    """Calculate the ETag for a local file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return '"' + hash_md5.hexdigest() + '"'


class CosemImage(BaseModel):
    name: str
    description: str
    url: str
    format: str
    grid_scale: list[float]
    grid_translation: list[float]
    grid_dims: list[str]
    grid_units: list[str]
    sample_type: str
    content_type: str

    @property
    def bucket_key(self) -> str:
        return self.url.split("janelia-cosem-datasets/", 1)[1]

    def download(self, path: str | PathLike, max_level: int | None = None) -> None:
        bucket = _bucket()
        dest = Path(path).expanduser().resolve()
        key = self.bucket_key
        with tqdm.tqdm(total=None) as pbar:
            for obj in bucket.objects.filter(Prefix=key):
                _dest = dest / str(obj.key)
                if _dest.exists() and calculate_etag(_dest) == obj.e_tag:
                    pbar.set_description(f"Already downloaded {obj.key}")
                    continue  # Skip this file because it's the same as the local file

                if max_level is not None:
                    # find the `/s{level}` in the key
                    match = re.search(r"\/s(\d+)\/", obj.key)
                    if match and int(match.group(1)) > max_level:
                        pbar.set_description(f"Skipping {obj.key}")
                        continue

                pbar.set_description(f"Downloading {obj.key}")
                _dest.parent.mkdir(parents=True, exist_ok=True)
                bucket.download_file(obj.key, str(_dest))

    def read(
        self, level: int | str = -1, transpose: Sequence[str] | None = None
    ) -> "DataTree":
        from microsim.cosem._tstore import read_tensorstore

        if isinstance(level, int):
            try:
                level = self.scales[level]
            except IndexError as e:
                raise IndexError(
                    f"Level {level!r} not found in {self.name!r}. "
                    f"Available levels are: {self.scales}"
                ) from e

        data = read_tensorstore(self, level=level)
        if transpose:
            data = data[ts.d[tuple(transpose)].transpose[:]]
        return data
        # from microsim.cosem._xarray import read_xarray
        # return read_xarray(self.url)  # type: ignore

    @property
    def attrs(self) -> dict[str, Any]:
        if not getattr(self, "_attrs", None):
            if self.format == "n5":
                attr = "/attributes.json"
            elif self.format == "zarr":
                attr = "/.zattrs"
            elif self.format == "precomputed":
                attr = "/info"
            self._attrs = fetch_s3_json(self.url + attr)
        return self._attrs

    @property
    def scales(self) -> list[str]:
        if not getattr(self, "_scales", None):
            scales = []
            # n5 multiscale
            if multi := self.attrs.get("multiscales"):
                for scale in multi:
                    if dsets := scale.get("datasets"):
                        for dset in dsets:
                            if path := dset.get("path"):
                                scales.append(path)
            # precomputed multiscale
            elif multi := self.attrs.get("scales"):
                for scale in multi:
                    if key := scale.get("key"):
                        scales.append(key)
            self._scales = scales
        return self._scales

    def show(self, **read_kwargs: Any) -> None:
        from pymmcore_widgets._stack_viewer_v2 import StackViewer
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if not (hadapp := app is not None):
            app = QApplication([])

        data = self.read(**read_kwargs)
        s = StackViewer(data)
        s.show()

        if not hadapp:
            app.exec()


class CosemDataset(BaseModel):
    name: str
    description: str
    thumbnail_url: str
    sample: CosemSample
    image_acquisition: CosemImageAcquisition
    images: list[CosemImage]

    @computed_field(repr=False)  # type: ignore [misc]
    @property
    def views(self) -> list["CosemView"]:
        return [v for v in fetch_views() if v.dataset_name == self.name]

    def view(self, name: str) -> "CosemView":
        return next(v for v in self.views if v.name == name)

    def image(self, **kwargs: Any) -> "CosemImage":
        return next(
            i for i in self.images if all(getattr(i, k) == v for k, v in kwargs.items())
        )

    @property
    def em_layers(self) -> list[CosemImage]:
        return [i for i in self.images if i.content_type == "em"]

    @property
    def segmentation_layers(self) -> list[CosemImage]:
        """Return list of all segmentation layers in the dataset.

        These are predictions that have undergone refinements such as thresholding,
        smoothing, size filtering, and connected component analysis.
        """
        return [i for i in self.images if i.content_type == "segmentation"]

    @property
    def prediction_layers(self) -> list[CosemImage]:
        """Return list of all prediction layers in the dataset.

        Raw distance transform inferences scaled from 0 to 255.
        A voxel value of 127 represent a predicted distance of 0 nm.
        """
        return [i for i in self.images if i.content_type == "prediction"]

    @property
    def analysis_layers(self) -> list[CosemImage]:
        """Return list of all analysis layers in the dataset.

        These are images that have undergone post-processing and analysis.
        """
        return [i for i in self.images if i.content_type == "analysis"]

    @property
    def thumbnail(self) -> "np.ndarray":
        from imageio.v3 import imread

        return imread(self.thumbnail_url)

    @classmethod
    def names(cls) -> list[str]:
        """Return list of all available dataset names."""
        return list(fetch_datasets())

    @classmethod
    def fetch(cls, name: str) -> "CosemDataset":
        """Fetch dataset with a specific name."""
        return fetch_datasets()[name]

    def read(self, image_keys: str | Sequence[str], **read_kwargs: Any) -> "DataTree":
        """Return DataTree for `image_key`."""
        if not image_keys:
            raise ValueError("No image keys provided")
        if isinstance(image_keys, str):
            image = self.image(name=image_keys)
            return image.read()
        if isinstance(image_keys, Sequence):
            images = [self.image(name=k) for k in image_keys]
            layers = [i.read(**read_kwargs).astype(ts.uint16) for i in images]
            return ts.stack(layers)
        raise ValueError(f"image_keys must be a str or list of str, not {image_keys!r}")

    def show(self, image_keys: str | Sequence[str], **read_kwargs: Any) -> None:
        from pymmcore_widgets._stack_viewer_v2 import StackViewer
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if not (hadapp := app is not None):
            app = QApplication([])

        data = self.read(image_keys, **read_kwargs)
        s = StackViewer(data)
        s.show()

        if not hadapp:
            app.exec()


class CosemTaxon(BaseModel):
    name: str
    short_name: str


class CosemView(BaseModel):
    name: str
    dataset_name: str
    description: str
    thumbnail_url: str
    position: list[float] | None
    scale: float | None
    orientation: str | list[int] | None
    created_at: datetime.datetime
    taxa: list[CosemTaxon]
    images: list[CosemImage]

    def load(self) -> "xr.DataArray": ...


if __name__ == "__main__":
    from rich import print

    # print(CosemDataset.fetch("jrc_hela-2").thumbnail)
    print(organelles())
