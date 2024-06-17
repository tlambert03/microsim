from __future__ import annotations

import logging
from contextlib import suppress

try:
    import boto3
    from botocore import UNSIGNED, client
    from supabase import Client
except ImportError as e:
    raise ImportError("To use cosem data, please `pip install microsim[cosem]`") from e
else:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

import hashlib
import re
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, get_args

import tqdm
from pydantic import BaseModel

from microsim.util import microsim_cache

try:
    from rich import print
except ImportError:
    pass

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from os import PathLike

    import botocore.client
    import botocore.response
    import supabase
    from mypy_boto3_s3 import S3Client

    from .models import CosemDataset, CosemView


COSEM_BUCKET = "janelia-cosem-datasets"
COSEM_CACHE = microsim_cache() / COSEM_BUCKET
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
    return Client(url, key)


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
        anno: Any = info.annotation
        with suppress(TypeError):
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                anno = args[0]

        try:
            is_model = isinstance(anno, type) and issubclass(anno, BaseModel)
        except TypeError:
            is_model = False

        if is_model:
            name = anno.__name__
            if name.startswith("Cosem"):
                name = name[5:]
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            yield (f"{field}:{name}", _collect_fields(anno))
        else:
            yield field


T = TypeVar("T", bound=BaseModel)


def fetch_all(type_: type[T]) -> tuple[T, ...]:
    table_name = type_.__name__.lower().replace("cosem", "")
    query = model_query(type_)
    try:
        response = _supabase().from_(table_name).select(query).execute()
    except Exception as e:
        raise ValueError(
            f"Failed to fetch {table_name!r} from Supabase. See above for details."
        ) from e
    return tuple(type_.model_validate(x) for x in response.data)


@cache
def fetch_datasets() -> Mapping[str, CosemDataset]:
    """Fetch all dataset metadata from the COSEM database."""
    from .models import CosemDataset

    return MappingProxyType({d.name: d for d in fetch_all(CosemDataset)})


@cache
def fetch_views() -> tuple[CosemView, ...]:
    """Fetch all view metadata from the COSEM database."""
    from .models import CosemView

    return fetch_all(CosemView)


@cache
def organelles() -> Mapping[str, list[CosemView]]:
    """Return a mapping of organelle names to their descriptions."""
    orgs: defaultdict[str, list[CosemView]] = defaultdict(list)
    for view in fetch_views():
        for taxon in view.taxa:
            orgs[taxon.name].append(view)
    return MappingProxyType(orgs)


# pattern to look for scale levels in the bucket keys as /s{level}/
SCALE_RE = re.compile(r"\/s(\d+)\/")


def _keys_tags(  # pragma: no cover
    prefix: str, max_level: int | None = 0, bucket_name: str = COSEM_BUCKET
) -> Iterator[tuple[str, str]]:
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            # exclude keys with a scale level greater than max_level
            if max_level is not None:
                if match := SCALE_RE.search(obj["Key"]):
                    if int(match.group(1)) > max_level:
                        continue
            yield obj["Key"], obj["ETag"]


def download_bucket_path(  # pragma: no cover
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
        for key, etag in _keys_tags(bucket_key, max_level=max_level)
    ]

    # download the files concurrently
    print(f"Downloading {bucket_key} to {dest}")
    with ThreadPoolExecutor(max_workers=MAX_CONNECTIONS) as executor:
        list(tqdm.tqdm(executor.map(_download_file, items), total=len(items)))


def _download_file(item: tuple[str, str, Path, str]) -> None:  # pragma: no cover
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
