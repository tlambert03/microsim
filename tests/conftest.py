import shutil
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

from microsim import util
from microsim.schema.settings import BackendName

AVAILABLE_BACKENDS = ["numpy"]
with suppress(ImportError):
    import cupy  # noqa

    AVAILABLE_BACKENDS.append("cupy")
with suppress(ImportError):
    import jax  # noqa

    AVAILABLE_BACKENDS.append("jax")
# with suppress(ImportError):
#     import torch

#     AVAILABLE_BACKENDS.append("torch")


@pytest.fixture(params=AVAILABLE_BACKENDS)
def np_backend(request: pytest.FixtureRequest) -> BackendName:
    return request.param  # type: ignore


@pytest.fixture(autouse=True, scope="session")
def fake_cache() -> Iterator[Path]:
    test_cache = Path(tempfile.mkdtemp(), "test_cache")
    test_cache.mkdir(parents=True, exist_ok=True)
    util._MICROSIM_CACHE = test_cache
    try:
        yield test_cache
    finally:
        shutil.rmtree(test_cache.parent, ignore_errors=True)
