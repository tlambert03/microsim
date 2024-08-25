"""Models for COSEM data."""

import datetime
import json
import logging
import urllib.parse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, computed_field

from microsim.util import ndview, norm_name

from ._client import (
    COSEM_BUCKET,
    download_bucket_path,
    fetch_datasets,
    fetch_s3,
    fetch_views,
)
from ._tstore import BinMode

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
    """Information about the sample used in the COSEM dataset."""

    name: str
    description: str
    protocol: str
    type: list[str] | None
    subtype: list[str] | None
    organism: list[str] | None


class CosemImageAcquisition(BaseModel):
    """Information about the image acquisition process."""

    name: str
    start_date: datetime.datetime | None
    grid_axes: list[str]
    grid_dimensions: list[float]
    grid_dimensions_unit: str
    grid_spacing: list[float]
    grid_spacing_unit: str


class CosemImage(BaseModel):
    """Pointer to an image in the COSEM dataset.

    An image contains all levels of the image pyramid and metadata.
    """

    name: str
    description: str
    url: Annotated[str, AfterValidator(lambda x: x.rstrip("/"))]
    format: ImageFormat
    grid_scale: list[float]
    grid_translation: list[float]
    grid_dims: list[str]
    grid_units: list[str]
    sample_type: SampleType
    content_type: ContentType
    created_at: datetime.datetime
    dataset_name: str
    # display_settings: dict
    # grid_index_order: str  # C-order or Fortran-order
    # id: int
    # institution: str
    # source: dict | None
    stage: Literal["prod", "dev"]

    @property
    def bucket_path(self) -> str:
        return urllib.parse.urlparse(self.url).path

    @computed_field  # type: ignore
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
        bin_mode: Literal[BinMode, "auto"] = "auto",
    ) -> "TensorStore":
        """Read image as TensorStore.

        Parameters
        ----------
        level : int
            Level of the image pyramid to read. Negative indices may be used to
            specify up from the the lowest resolution.
            Default is -1 (lowest resolution).
        transpose : tuple
            Axes to transpose the image to.  Default is ("y", "x", "z"), which orients
            most images with the coverslip co-planar with the last two dimensions.
        bin_mode : {"standard", "sum", "auto"}
            Method to bin the image.  Options are "standard" or "sum".  Default is
            "auto", which chooses "sum" for segmentation images and "standard" for other
            images.  In COSEM, the "standard" binning method is to take the statistical
            mode of the pixels in each bin.  This is useful for maintaining instance
            segmentation IDs, but is not good for simulation purposes, so "sum" is used
            when loading segmentation images.
        """
        from microsim.cosem._tstore import read_tensorstore

        if bin_mode == "auto":
            # we convert segmentation images to sum mode, because they are encoded
            # for instance segmentation, which is not what we want.
            bin_mode = "sum" if self.content_type == "segmentation" else "standard"
        return read_tensorstore(
            self, level=level, transpose=transpose, bin_mode=bin_mode
        )

    def read_xarray(self) -> "xr.DataArray | DataTree":
        """Read image as xarray or DataTree.

        This is less tested and used than `read`.  Let me know if you need it.
        """
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
            try:
                self._attrs = json.load(fetch_s3(self.url + attr))
            except Exception:
                self._attrs = {}
        return self._attrs  # type: ignore [no-any-return]

    def show(self, **read_kwargs: Any) -> None:
        ndview(self.read(**read_kwargs))


class CosemDataset(BaseModel):
    """Top-level container for a COSEM dataset."""

    name: str
    description: str
    thumbnail_url: str
    sample: CosemSample
    image_acquisition: CosemImageAcquisition
    images: list[CosemImage]
    created_at: datetime.datetime

    @computed_field(repr=False)  # type: ignore [prop-decorator]
    @property
    def views(self) -> list["CosemView"]:
        """Return list of all views in the dataset."""
        return [v for v in fetch_views() if v.dataset_name == self.name]

    @classmethod
    def fetch(cls, name: str) -> "CosemDataset":
        """Fetch dataset with a specific name."""
        datasets = fetch_datasets()
        if name not in datasets:
            if key := _get_similar(name, datasets):
                logging.warning(
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
        """Return a mapping of all available cosem datasets."""
        return fetch_datasets()

    @classmethod
    def names(cls) -> list[str]:
        """Return list of all available dataset names."""
        return sorted(cls.all())

    def view(self, name: str) -> "CosemView":
        """Return view with a specific name."""
        return next(v for v in self.views if v.name == name)

    def image(self, **kwargs: Any) -> "CosemImage":
        """Return first available image matching kwargs.

        Examples
        --------
        >>> dataset.image(name="er_seg")
        """
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
        """Return list of all EM layers in the dataset."""
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
        """Return thumbnail image as numpy array."""
        try:
            from imageio.v3 import imread
        except ImportError as e:
            raise ImportError(
                "To use the thumbnail property, install the imageio package."
            ) from e

        return imread(self.thumbnail_url)  # type: ignore [no-any-return]

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
            return image.read(**read_kwargs)
        if isinstance(image_keys, Sequence):
            import tensorstore as ts

            images = [self.image(name=k) for k in image_keys]
            layers = [i.read(**read_kwargs).astype(ts.uint16) for i in images]
            return ts.stack(layers)
        raise ValueError(  # pragma: no cover
            f"image_keys must be a str or list of str, not {image_keys!r}"
        )

    def show(self, image_keys: str | Sequence[str], **read_kwargs: Any) -> None:
        """Show images in `image_keys` using ndview."""
        data = self.read(image_keys, **read_kwargs)
        ndview(data)


class CosemTaxon(BaseModel):
    """Organelle taxonomy information."""

    name: str
    short_name: str


class CosemView(BaseModel):
    """A pre-defined view of a dataset."""

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
        """Return all views matching specific kwargs.

        Examples
        --------
        >>> CosemView.filter(dataset_name="jrc_hela-3")
        """
        matches = []
        for view in fetch_views():
            if all(getattr(view, k) == v for k, v in kwargs.items()):
                matches.append(view)
        return matches

    @classmethod
    def all(cls) -> tuple["CosemView", ...]:
        """Return all available views."""
        return fetch_views()


def _get_similar(search_term: str, available: Iterable[str]) -> str | None:
    """Return similar string from available list, if any."""
    for avail in available:
        if norm_name(search_term) == norm_name(avail):
            return avail
    return None
