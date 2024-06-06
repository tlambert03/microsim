import shutil
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

import microsim.schema as ms
from microsim import util
from microsim.schema.optical_config.lib import FITC
from microsim.schema.settings import BackendName

TRUTH_SPACE = ms.ShapeScaleSpace(shape=(64, 128, 128), scale=(0.04, 0.02, 0.02))
MATSLINES = ms.MatsLines(density=0.4, length=30, azimuth=5, max_r=1)
GREEN_MATSLINES = ms.FluorophoreDistribution(distribution=MATSLINES)
NA1_4 = ms.ObjectiveLens(numerical_aperture=1.4)


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


@pytest.fixture(autouse=True, scope="module")
def fake_cache() -> Iterator[Path]:
    test_cache = Path(tempfile.mkdtemp(), "test_cache")
    test_cache.mkdir(parents=True, exist_ok=True)
    util._MICROSIM_CACHE = test_cache
    try:
        yield test_cache
    finally:
        shutil.rmtree(test_cache.parent, ignore_errors=True)


@pytest.fixture
def sim1() -> ms.Simulation:
    return ms.Simulation(
        truth_space=TRUTH_SPACE,
        output_space={"downscale": 4},
        sample=ms.Sample(labels=[GREEN_MATSLINES]),
        detector=ms.CameraCCD(qe=0.82, read_noise=6, bit_depth=12),
        objective_lens=NA1_4,
        channels=[FITC],
    )


@pytest.fixture(autouse=True)
def mpl_show_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    with suppress(ImportError):
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)
