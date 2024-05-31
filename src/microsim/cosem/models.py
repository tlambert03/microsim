"""Models for COSEM data.

Dataset names as of May 31, 2024:
aic_desmosome-1
aic_desmosome-2
aic_desmosome-3
jrc_ccl81-covid-1
jrc_choroid-plexus-2
jrc_cos7-11
jrc_cos7-1a
jrc_cos7-1b
jrc_ctl-id8-1
jrc_ctl-id8-2
jrc_ctl-id8-3
jrc_ctl-id8-4
jrc_ctl-id8-5
jrc_dauer-larva
jrc_fly-acc-calyx-1
jrc_fly-ellipsoid-body
jrc_fly-fsb-1
jrc_fly-fsb-2
jrc_fly-larva-1
jrc_fly-mb-z0419-20
jrc_fly-protocerebral-bridge
jrc_fly-vnc-1
jrc_hela-1
jrc_hela-2
jrc_hela-21
jrc_hela-22
jrc_hela-3
jrc_hela-4
jrc_hela-bfa
jrc_hela-h89-1
jrc_hela-h89-2
jrc_hela-nz-1
jrc_jurkat-1
jrc_macrophage-2
jrc_mus-dorsal-striatum
jrc_mus-epididymis-1
jrc_mus-epididymis-2
jrc_mus-granule-neurons-1
jrc_mus-granule-neurons-2
jrc_mus-granule-neurons-3
jrc_mus-guard-hair-follicle
jrc_mus-heart-1
jrc_mus-hippocampus-1
jrc_mus-kidney
jrc_mus-kidney-2
jrc_mus-kidney-3
jrc_mus-liver
jrc_mus-liver-2
jrc_mus-liver-3
jrc_mus-meissner-corpuscle-1
jrc_mus-meissner-corpuscle-2
jrc_mus-nacc-2
jrc_mus-nacc-3
jrc_mus-nacc-4
jrc_mus-pacinian-corpuscle
jrc_mus-pancreas-1
jrc_mus-pancreas-2
jrc_mus-pancreas-3
jrc_mus-pancreas-4
jrc_mus-sc-zp104a
jrc_mus-sc-zp105a
jrc_mus-skin-1
jrc_mus-thymus-1
jrc_sum159-1
"""

import datetime
import json
import logging
import urllib.parse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, computed_field

from microsim.util import norm_name

from ._client import (
    COSEM_BUCKET,
    download_bucket_path,
    fetch_datasets,
    fetch_s3,
    fetch_views,
)

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from datatree import DataTree
    from tensorstore import TensorStore

# ------------------------ MODELS ------------------------


SampleType = Literal["scalar", "label", "geometry"]
ContentType = Literal[
    "lm", "em", "segmentation", "prediction", "analysis", "annotation", "mesh"
]
ImageFormat = Literal[
    "n5",
    "zarr",
    "precomputed",
    "neuroglancer_precomputed",
    "neuroglancer_multilod_draco",
    "neuroglancer_legacy_mesh",
]


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
    grid_dimensions: list[float]
    grid_dimensions_unit: str
    grid_spacing: list[float]
    grid_spacing_unit: str


class CosemImage(BaseModel):
    name: str
    description: str
    url: str
    format: ImageFormat
    grid_scale: list[float]
    grid_translation: list[float]
    grid_dims: list[str]
    grid_units: list[str]
    sample_type: SampleType
    content_type: ContentType
    created_at: datetime.datetime
    # dataset_name: str
    # display_settings: dict
    # grid_index_order: str  # C-order or Fortran-order
    # id: int
    # institution: str
    # source: dict | None
    # stage: Literal["prod", "dev"]

    @property
    def bucket_path(self) -> str:
        return urllib.parse.urlparse(self.url).path

    @computed_field
    @property
    def scales(self) -> list[str]:
        """Fetch all available scales for the image from s3."""
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

    @property
    def bucket_key(self) -> str:
        return self.url.split(f"{COSEM_BUCKET}/", 1)[1]

    def download(
        self, dest: str | PathLike | None = None, max_level: int | None = 0
    ) -> None:
        return download_bucket_path(self.bucket_key, dest=dest, max_level=max_level)

    def read(
        self,
        level: int = -1,
        transpose: Sequence[str] | None = ("y", "x", "z"),
        # bin_mode: Literal["mode", "sum"] = "mode",
    ) -> "TensorStore":
        from microsim.cosem._tstore import read_tensorstore

        return read_tensorstore(self, level=level, transpose=transpose)

    def read_xarray(self) -> "xr.DataArray | DataTree":
        from microsim.cosem._xarray import read_xarray

        return read_xarray(self.url)

    @property
    def attrs(self) -> dict[str, Any]:
        if not getattr(self, "_attrs", None):
            if self.format == "n5":
                attr = "/attributes.json"
            elif self.format == "zarr":
                attr = "/.zattrs"
            elif self.format == "precomputed":
                attr = "/info"
            self._attrs = json.load(fetch_s3(self.url + attr))
        return self._attrs  # type: ignore [no-any-return]

    def show(self, **read_kwargs: Any) -> None:
        from microsim.util import view_nd

        view_nd(self.read(**read_kwargs))


class CosemDataset(BaseModel):
    name: str
    description: str
    thumbnail_url: str
    sample: CosemSample
    image_acquisition: CosemImageAcquisition
    images: list[CosemImage]
    created_at: datetime.datetime

    @computed_field(repr=False)  # type: ignore [misc]
    @property
    def views(self) -> list["CosemView"]:
        return [v for v in fetch_views() if v.dataset_name == self.name]

    @classmethod
    def fetch(cls, name: str) -> "CosemDataset":
        """Fetch dataset with a specific name."""
        datasets = fetch_datasets()
        if name not in datasets:
            if key := _get_similar(name, datasets):
                logging.warn(
                    f"Dataset {name!r} not found. Using similar {key!r} instead."
                )
                name = key
            else:
                raise ValueError(
                    f"Dataset {name!r} not found. Available datasets: {cls.names()}`"
                )
        return datasets[name]

    @classmethod
    def all(cls) -> Mapping[str, "CosemDataset"]:
        """Fetch dataset with a specific name."""
        return fetch_datasets()

    @classmethod
    def names(cls) -> list[str]:
        """Return list of all available dataset names."""
        return sorted(cls.all())

    def view(self, name: str) -> "CosemView":
        return next(v for v in self.views if v.name == name)

    def image(self, **kwargs: Any) -> "CosemImage":
        avail: defaultdict[str, set[str]] = defaultdict(set)
        for img in self.images:
            if all(getattr(img, k) == v for k, v in kwargs.items()):
                return img
            for key in kwargs:
                avail[key].add(getattr(img, key))

        _avail = {k: sorted(v) for k, v in avail.items()}
        raise ValueError(f"Image not found with {kwargs!r}.  Available keys: {_avail}")

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

    def read(
        self, image_keys: str | Sequence[str], **read_kwargs: Any
    ) -> "TensorStore":
        """Return TensorStore for `image_keys`.

        This reads all images in `image_keys` and stacks them along the first axis.
        """
        if not image_keys:
            raise ValueError("No image keys provided")
        if isinstance(image_keys, str):
            image = self.image(name=image_keys)
            return image.read()
        if isinstance(image_keys, Sequence):
            import tensorstore as ts

            images = [self.image(name=k) for k in image_keys]
            layers = [i.read(**read_kwargs).astype(ts.uint16) for i in images]
            return ts.stack(layers)
        raise ValueError(  # pragma: no cover
            f"image_keys must be a str or list of str, not {image_keys!r}"
        )

    def show(self, image_keys: str | Sequence[str], **read_kwargs: Any) -> None:
        from microsim.util import view_nd

        data = self.read(image_keys, **read_kwargs)
        view_nd(data)


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
    orientation: list[int] | None
    created_at: datetime.datetime
    taxa: list[CosemTaxon]
    images: list[CosemImage]

    @classmethod
    def filter(cls, **kwargs: Any) -> list["CosemView"]:
        """Fetch dataset with a specific name."""
        matches = []
        for view in fetch_views():
            if all(getattr(view, k) == v for k, v in kwargs.items()):
                matches.append(view)
        return matches

    @classmethod
    def all(cls) -> tuple["CosemView", ...]:
        """Fetch dataset with a specific name."""
        return fetch_views()


class CosemImagery(BaseModel):
    content_type: str
    coordinate_space: str
    created_at: str
    dataset_name: str
    description: str
    display_settings: dict
    format: str
    grid_dims: list[str]
    grid_index_order: str
    grid_scale: list[int]
    grid_translation: list[int]
    grid_units: list[str]
    id: int
    institution: str
    name: str
    sample_type: str
    source: dict
    stage: str
    url: str


def _get_similar(search_term: str, available: Iterable[str]) -> str | None:
    for avail in available:
        if norm_name(search_term) == norm_name(avail):
            return avail
    return None
