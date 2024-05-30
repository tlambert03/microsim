from __future__ import annotations

import hashlib
import logging
import re
import shutil
import urllib.request
from collections import defaultdict
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import botocore.response
import tqdm

from microsim.util import MICROSIM_CACHE

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    import botocore.client
    import botocore.response
    import supabase
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.service_resource import Bucket

    from .models import CosemDataset, CosemView


COSEM_BUCKET = "janelia-cosem-datasets"
COSEM_CACHE = MICROSIM_CACHE / COSEM_BUCKET


def bucket_cache() -> Path:
    return COSEM_CACHE


def clear_cache() -> None:
    shutil.rmtree(COSEM_CACHE, ignore_errors=True)


@cache
def _supabase() -> supabase.Client:
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
def _s3_client() -> S3Client:
    logging.getLogger("botocore").setLevel(logging.WARNING)
    import boto3
    from botocore import UNSIGNED, client

    return boto3.client("s3", config=client.Config(signature_version=UNSIGNED))


@cache
def _bucket(name: str = COSEM_BUCKET) -> Bucket:
    import boto3
    from botocore import UNSIGNED, client

    session = boto3.Session()
    resource = session.resource("s3", config=client.Config(signature_version=UNSIGNED))
    return resource.Bucket(name)


def fetch_s3(url: str) -> botocore.response.StreamingBody:
    proto, _, bucket, key = url.split("/", 3)
    if not proto.startswith("s3"):
        raise ValueError(f"Unsupported protocol {proto!r}")

    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    response_meta = obj["ResponseMetadata"]
    if not (status := response_meta.get("HTTPStatusCode")) == 200:
        raise ValueError(f"Failed to fetch {url!r} with status {status}")
    return obj["Body"]


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
def fetch_datasets() -> Mapping[str, CosemDataset]:
    """Fetch all dataset metadata from the COSEM database."""
    from .models import CosemDataset

    response = _supabase().from_("dataset").select(DATASETS_QUERY).execute()
    datasets: dict[str, CosemDataset] = {}
    for x in response.data:
        ds = CosemDataset.model_validate(x)
        datasets[ds.name] = ds
    return MappingProxyType(datasets)


@cache
def fetch_views() -> list[CosemView]:
    """Fetch all view metadata from the COSEM database."""
    from .models import CosemView

    response = _supabase().from_("view").select(VIEWS_QUERY).execute()
    return [CosemView.model_validate(x) for x in response.data]


@cache
def organelles() -> Mapping[str, str]:
    orgs: defaultdict[str, list[CosemView]] = defaultdict(list)
    for view in fetch_views():
        for taxon in view.taxa:
            orgs[taxon.name].append(view)
    return MappingProxyType(orgs)


def calculate_etag(file_path: str | Path) -> str:
    """Calculate the ETag for a local file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return '"' + hash_md5.hexdigest() + '"'


def download_bucket_path(
    bucket_key: str, dest: str | PathLike | None = None, max_level: int | None = 0
) -> None:
    """Download a bucket path to a local `dest` directory.

    Previously downloaded files are skipped if the ETag matches the remote file.
    If not specified, the destination directory is the cache directory.

    Parameters
    ----------
    bucket_key : str
        The key of the bucket path to download.
    dest : str | PathLike | None
        The destination directory to download the bucket path to. Will be created if it
        does not exist. If `None` (the default), the cache directory is used.
    max_level : int | None
        The maximum image scale level to download. If None, all levels are downloaded.
        By default, only the highest resolution level is downloaded.
    """
    if dest is None:
        dest = COSEM_CACHE

    bucket = _bucket()
    dest = Path(dest).expanduser().resolve()
    with tqdm.tqdm(total=None) as pbar:
        for obj in bucket.objects.filter(Prefix=bucket_key):
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
        pbar.set_description(f"Downloaded {bucket_key} to {dest}")


if __name__ == "__main__":
    from rich import print

    # print(CosemDataset.fetch("jrc_hela-2").thumbnail)
    print(organelles())
