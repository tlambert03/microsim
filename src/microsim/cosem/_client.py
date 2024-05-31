from __future__ import annotations

import hashlib
import logging
import re
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, get_args

import tqdm
from pydantic import BaseModel

try:
    import boto3
    from botocore import UNSIGNED, client
    from supabase import Client
except ImportError as e:
    raise ImportError("To use cosem data, please `pip install microsim[cosem]`") from e
else:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

from rich import print

from microsim.util import MICROSIM_CACHE

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from os import PathLike

    import botocore.client
    import botocore.response
    import supabase
    from mypy_boto3_s3 import S3Client

    from .models import CosemDataset, CosemView


COSEM_BUCKET = "janelia-cosem-datasets"
COSEM_CACHE = MICROSIM_CACHE / COSEM_BUCKET
MAX_CONNECTIONS = 50
CFG = client.Config(signature_version=UNSIGNED, max_pool_connections=MAX_CONNECTIONS)


def bucket_cache() -> Path:
    return COSEM_CACHE


def clear_cache() -> None:
    shutil.rmtree(COSEM_CACHE, ignore_errors=True)


@cache
def _supabase(url: str | None = None, key: str | None = None) -> supabase.Client:
    if not (url and key):
        with urllib.request.urlopen(
            "https://openorganelle.janelia.org/static/js/4743.a9f85e14.chunk.js"
        ) as response:
            if response.status != 200:
                raise ValueError("Failed to fetch Supabase URL and key")
            text = response.read().decode("utf-8")
        key = text.split("SUPABASE_KEY:")[1].split(",")[0].strip("\"'")
        url = text.split("SUPABASE_URL:")[1].split(",")[0].strip("\"'")
    return Client(url, key)  # type: ignore


@cache
def _s3_client() -> S3Client:
    return boto3.client("s3", config=CFG)


def fetch_s3(url: str) -> botocore.response.StreamingBody:
    proto, _, bucket, key = url.split("/", 3)
    if not proto.startswith("s3"):
        raise ValueError(f"Unsupported protocol {proto!r}")

    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    response_meta = obj["ResponseMetadata"]
    if not (status := response_meta.get("HTTPStatusCode")) == 200:
        raise ValueError(f"Failed to fetch {url!r} with status {status}")
    return obj["Body"]


def model_query(model: type[BaseModel]) -> str:
    """Create a query string for fetching a model from Supabase."""
    result = []
    for item in _collect_fields(model):
        if isinstance(item, str):
            result.append(item)
        else:
            section_name, fields = item
            section_str = f"{section_name}({','.join(fields)})"
            result.append(section_str)
    return ",".join(result)


def _collect_fields(model: type[BaseModel]) -> Iterator[str | tuple]:
    """Used in model_query to recursively collect fields from a model."""
    for field, info in model.model_fields.items():
        args = get_args(info.annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            anno: Any = args[0]
        else:
            anno = info.annotation
        if isinstance(anno, type) and issubclass(anno, BaseModel):
            name = anno.__name__
            if name.startswith("Cosem"):
                name = name[5:]
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            yield (f"{field}:{name}", _collect_fields(anno))
        else:
            yield field


@cache
def fetch_datasets() -> Mapping[str, CosemDataset]:
    """Fetch all dataset metadata from the COSEM database."""
    from .models import CosemDataset

    query = model_query(CosemDataset)
    response = _supabase().from_("dataset").select(query).execute()
    datasets: dict[str, CosemDataset] = {}
    for x in response.data:
        ds = CosemDataset.model_validate(x)
        datasets[ds.name] = ds
    return MappingProxyType(datasets)


@cache
def fetch_views() -> tuple[CosemView, ...]:
    """Fetch all view metadata from the COSEM database."""
    from .models import CosemView

    query = model_query(CosemView)
    response = _supabase().from_("view").select(query).execute()
    return tuple(CosemView.model_validate(x) for x in response.data)


@cache
def organelles() -> Mapping[str, list[CosemView]]:
    """Return a mapping of organelle names to their descriptions."""
    orgs: defaultdict[str, list[CosemView]] = defaultdict(list)
    for view in fetch_views():
        for taxon in view.taxa:
            orgs[taxon.name].append(view)
    return MappingProxyType(orgs)


SCALE_RE = re.compile(r"\/s(\d+)\/")


def keys_tags(
    prefix: str, max_level: int | None = 0, bucket_name: str = COSEM_BUCKET
) -> Iterator[tuple[str, str]]:
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            if max_level is not None:
                # exclude keys with a scale level greater than max_level
                match = SCALE_RE.search(obj["Key"])
                if match and int(match.group(1)) > max_level:
                    continue
            yield obj["Key"], obj["ETag"]


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

    dest = Path(dest).expanduser().resolve()

    # Prepare the items for the _download_file function
    items = [
        (key, etag, dest, COSEM_BUCKET)
        for key, etag in keys_tags(bucket_key, max_level=max_level)
    ]

    # download the files concurrently
    print(f"Downloading {bucket_key} to {dest}")
    with ThreadPoolExecutor(max_workers=MAX_CONNECTIONS) as executor:
        list(tqdm.tqdm(executor.map(_download_file, items), total=len(items)))


def _download_file(item: tuple[str, str, Path, str]) -> None:
    key, etag, dest, bucket_name = item
    _dest = dest / str(key)
    if _dest.exists() and _calculate_etag(_dest) == etag:
        return
    _dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _s3_client().download_file(bucket_name, key, str(_dest))
    except Exception as e:
        logging.error(f"Failed to download {key!r}: {e}")


def _calculate_etag(file_path: str | Path) -> str:
    """Calculate the ETag for a local file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return '"' + hash_md5.hexdigest() + '"'
