from contextlib import suppress
from typing import Any

import numpy as np
import pytest
import scipy.fft
from scipy.signal import fftconvolve

from microsim.fft import convolve
from microsim.fft.backends import JaxFFTBackend, TorchFFTBackend
from microsim.fft.convolve import fftconvolve_same_pairs, patched_fftconvolve
from microsim.schema.backend import NumpyAPI

np.random.seed(0)
SHAPE = (128, 128, 128)
ARY = np.random.rand(*SHAPE).astype(np.float32)
KERNEL = np.random.randint(0, 200, (3, 3, 3)).astype(np.uint8)
EXPECTED = fftconvolve(ARY, KERNEL, mode="same")
BACKENDS: dict = {"scipy": "scipy"}
with suppress(ImportError):
    BACKENDS["jax"] = JaxFFTBackend()
with suppress(ImportError):
    BACKENDS["torch"] = TorchFFTBackend()


@pytest.mark.parametrize("backend", BACKENDS)
def test_fft_backend(backend: Any) -> None:
    with scipy.fft.set_backend(BACKENDS[backend], coerce=True):
        result = patched_fftconvolve(ARY, KERNEL, mode="same")

    if hasattr(result, "cpu"):
        # torch tensor ... required before calling assert_allclose
        result = result.cpu()

    # in many case this is accurate to 1e-6 tolerance, but it's os-dependent
    # so we set the tolerance to 1e-5
    TOL = 1e-5
    np.testing.assert_allclose(result, EXPECTED, rtol=TOL)


def test_fftconvolve_same_pairs() -> None:
    """Matches `fftconvolve(mode="same")` for every image/kernel pair."""
    rng = np.random.default_rng(0)
    images = [rng.random((7, 10, 11)), rng.random((7, 10, 11))]
    # varied kernel sizes (smallest first, one larger than the images), so that the
    # shared FFT shape must fit them all; and (1, 1) has no kernel at all.
    kernels = {
        (0, 0): ("a", rng.random((3, 3, 3))),
        (0, 1): ("b", rng.random((12, 13, 9))),
        (1, 0): ("c", rng.random((4, 9, 6))),
    }
    convolve._KERNEL_FT_CACHE.clear()
    for _ in range(2):  # second time around, kernel transforms come from the cache
        out = fftconvolve_same_pairs(images, kernels, 2, NumpyAPI(), cache_mb=10)
        for (o, i), (_, kernel) in kernels.items():
            expect = fftconvolve(images[i], kernel, mode="same")
            np.testing.assert_allclose(out[o][i], expect, atol=1e-12)
        np.testing.assert_array_equal(out[1][1], 0)
        assert len(convolve._KERNEL_FT_CACHE) == 3
