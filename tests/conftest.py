from contextlib import suppress

import pytest

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
