import datetime
import json
from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, computed_field

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
    from tensorstore import TensorStore

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
        return self.url.split(f"{COSEM_BUCKET}/", 1)[1]

    def download(
        self, dest: str | PathLike | None = None, max_level: int | None = 0
    ) -> None:
        return download_bucket_path(self.bucket_key, dest=dest, max_level=max_level)

    def read(
        self, level: int = -1, transpose: Sequence[str] | None = None
    ) -> "TensorStore":
        from microsim.cosem._tstore import read_tensorstore

        return read_tensorstore(self, level=level, transpose=transpose)

    def read_xarray(self) -> "xr.DataArray":
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
        return self._attrs

    @property
    def scales(self) -> list[str]:
        # TODO: it would be nice if we didn't need to hit the network for this.
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
        from microsim.util import view_nd

        view_nd(self.read(**read_kwargs))


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

    def read(
        self, image_keys: str | Sequence[str], **read_kwargs: Any
    ) -> "TensorStore":
        """Return DataTree for `image_key`."""
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
        print(data)
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
    orientation: str | list[int] | None
    created_at: datetime.datetime
    taxa: list[CosemTaxon]
    images: list[CosemImage]
