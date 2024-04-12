import datetime
import logging
import urllib.request
from functools import cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np
    from datatree import DataTree
    from supabase import Client


@cache
def _client() -> "Client":
    try:
        from supabase import Client
    except ImportError as e:
        raise ImportError(
            "To download cosem data, please first `pip install supabase`"
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


DATASET_QUERY = """
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


@cache
def fetch_datasets() -> dict[str, "CosemDataset"]:
    """Fetch all dataset metadata from the COSEM database."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    response = _client().from_("dataset").select(DATASET_QUERY).execute()
    datasets: dict[str, CosemDataset] = {}
    for x in response.data:
        ds = CosemDataset.model_validate(x)
        datasets[ds.name] = ds
    return datasets


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

    def read(self, level: int = 0) -> "DataTree":
        from microsim.cosem._xarray import read_xarray

        return read_xarray(self.url)  # type: ignore


class CosemDataset(BaseModel):
    name: str
    description: str
    thumbnail_url: str
    sample: CosemSample
    image_acquisition: CosemImageAcquisition
    images: list[CosemImage]

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

    def read_image(self, image_key: str, level: int = 0) -> "DataTree":
        """Return DataTree for `image_key`."""
        try:
            image = next(i for i in self.images if i.name == image_key)
        except StopIteration as e:
            raise KeyError(f"Image {image_key!r} not found in {self.name!r}") from e
        return image.read()


if __name__ == "__main__":
    from rich import print

    print(CosemDataset.fetch("jrc_hela-2").thumbnail)
