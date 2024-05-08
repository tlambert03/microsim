from typing import Any

import numpy as np
import pytest
import scipy.fft
from scipy.signal import fftconvolve

from microsim.fft.backends import JaxFFTBackend, TorchFFTBackend
from microsim.fft.convolve import patched_fftconvolve

np.random.seed(0)
SHAPE = (128, 128, 128)
ARY = np.random.rand(*SHAPE).astype(np.float32)
KERNEL = np.ones((3, 3, 3)).astype(np.uint8)
EXPECTED = fftconvolve(ARY, KERNEL, mode="same")
BACKENDS = {"scipy": "scipy", "jax": JaxFFTBackend(), "torch": TorchFFTBackend()}


@pytest.mark.parametrize("backend", ["scipy", "jax", "torch"])
def test_fft_backend(backend: Any) -> None:
    with scipy.fft.set_backend(BACKENDS[backend], coerce=True):
        result = patched_fftconvolve(ARY, KERNEL, mode="same")

    if hasattr(result, "cpu"):
        # torch tensor ... required before calling assert_allclose
        result = result.cpu()

    np.testing.assert_allclose(result, EXPECTED, rtol=1e-6)
