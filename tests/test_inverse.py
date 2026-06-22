"""Fast checks that the differentiable forward model + inverse helpers work.

The full demonstrations live in `examples/inverse/`; here we run tiny versions
so the gradient machinery is exercised quickly in CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from microsim.inverse import fit, normalized_mse, poisson_nll, softplus_nonneg
from microsim.optics import vectorial_psf

EXAMPLES = Path(__file__).parent.parent / "examples" / "inverse"


def _small_psf(coeffs, ansi=(5, 12)):
    return vectorial_psf(
        nz=5,
        nx=32,
        dxy=0.06,
        dz=0.2,
        wvl=0.55,
        na=1.2,
        ni=1.515,
        ansi_indices=ansi,
        coefficients=coeffs,
        normalize="sum",
    )


def test_psf_is_differentiable() -> None:
    """Gradient of an image loss flows back to the Zernike coefficients."""
    grad = jax.grad(lambda c: _small_psf(c)[2].sum())(jnp.array([0.05, 0.0]))
    assert grad.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_phase_retrieval_recovers() -> None:
    """A short optimization moves the coefficients toward the truth."""
    true = jnp.array([0.06, 0.04])
    measured = _small_psf(true)

    def loss_fn(c):
        return normalized_mse(_small_psf(c), measured)

    result = fit(loss_fn, jnp.zeros(2), steps=80, learning_rate=5e-3)
    err = np.abs(np.asarray(result.params) - np.asarray(true))
    assert result.losses[-1] < result.losses[0]
    assert err.max() < 0.02  # within 20 nm RMS wavefront


def test_deconvolution_improves() -> None:
    """Gradient deconvolution gets closer to the truth than the blurred input."""
    from jax.scipy.signal import fftconvolve

    rng = np.random.default_rng(0)
    truth = np.zeros((48, 48), np.float32)
    pts = rng.integers(6, 42, size=(12, 2))
    truth[pts[:, 0], pts[:, 1]] = 1.0
    truth = jnp.asarray(truth)
    psf = vectorial_psf(nz=1, nx=21, dxy=0.06, dz=0.1, wvl=0.55, na=1.2, ni=1.515)[0]
    blurred = fftconvolve(truth, psf, mode="same")

    def loss_fn(raw):
        return poisson_nll(
            fftconvolve(softplus_nonneg(raw), psf, mode="same") * 500, blurred * 500
        )

    result = fit(loss_fn, jnp.full((48, 48), -4.0), steps=120, learning_rate=0.05)
    recon = np.asarray(softplus_nonneg(result.params))
    mse_blur = np.mean((np.asarray(blurred) - np.asarray(truth)) ** 2)
    mse_recon = np.mean((recon - np.asarray(truth)) ** 2)
    assert mse_recon < mse_blur


def test_example_modules_import() -> None:
    """The example scripts import cleanly (no execution of main())."""
    for name in ("phase_retrieval", "deconvolution", "adaptive_optics"):
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        assert spec and spec.loader
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
