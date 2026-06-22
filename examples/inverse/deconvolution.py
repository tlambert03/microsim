"""Sample estimation (deconvolution) by gradient descent.

Given a blurred, photon-noisy widefield image and a known PSF, recover the
underlying fluorophore distribution.  We optimize the (non-negative) sample
density to minimize a Poisson negative-log-likelihood plus a total-variation
prior - i.e. a regularized, differentiable form of Richardson-Lucy that drops
straight out of microsim's differentiable forward model.

Run:  python examples/inverse/deconvolution.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve

from microsim.inverse import fit, poisson_nll, softplus_nonneg, total_variation
from microsim.optics import vectorial_psf

NX = 96
PHOTONS = 200.0  # peak photons - deliberately photon-starved
TV_WEIGHT = 2e-3


def make_sample(n_points: int = 40, seed: int = 0) -> jax.Array:
    """A sparse field of bright point emitters (the unknown 'truth')."""
    rng = np.random.default_rng(seed)
    img = np.zeros((NX, NX), np.float32)
    pts = rng.integers(8, NX - 8, size=(n_points, 2))
    img[pts[:, 0], pts[:, 1]] = rng.uniform(0.4, 1.0, size=n_points)
    return jnp.asarray(img)


def make_psf() -> jax.Array:
    """Aberration-free focal-plane PSF of the imaging system."""
    return vectorial_psf(nz=1, nx=33, dxy=0.05, dz=0.1, wvl=0.55, na=1.3, ni=1.515)[0]


def simulate_image(sample: jax.Array, psf: jax.Array, seed: int = 1) -> jax.Array:
    """Blur the sample with the PSF and add Poisson photon noise."""
    blurred = fftconvolve(sample, psf, mode="same")
    counts = jax.random.poisson(jax.random.PRNGKey(seed), blurred * PHOTONS)
    return counts / PHOTONS


def deconvolve(measured: jax.Array, psf: jax.Array, steps: int = 300, lr: float = 0.05):
    """Recover the sample density from the measured image."""

    def forward(raw: jax.Array) -> jax.Array:
        return fftconvolve(softplus_nonneg(raw), psf, mode="same")

    def loss_fn(raw: jax.Array) -> jax.Array:
        pred = forward(raw)
        nll = poisson_nll(pred * PHOTONS, measured * PHOTONS)
        return nll + TV_WEIGHT * total_variation(softplus_nonneg(raw))

    init = jnp.full((NX, NX), -4.0)  # softplus(-4) ≈ 0, a near-empty start
    result = fit(loss_fn, init, steps=steps, learning_rate=lr)
    return softplus_nonneg(result.params), result.losses


def main() -> None:
    sample = make_sample()
    psf = make_psf()
    measured = simulate_image(sample, psf)
    recon, _ = deconvolve(measured, psf)

    t, m, r = (np.asarray(x) for x in (sample, measured, recon))
    mse_blur = np.mean((m - t) ** 2)
    mse_recon = np.mean((r - t) ** 2)
    print("Deconvolution / sample estimation:")
    print(f"  MSE blurred vs truth   : {mse_blur:.5f}")
    print(f"  MSE recovered vs truth : {mse_recon:.5f}")
    print(f"  improvement            : {mse_blur / mse_recon:.2f}x")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in [
        (axs[0], t, "true sample"),
        (axs[1], m, "measured (blurred + noisy)"),
        (axs[2], r, "recovered"),
    ]:
        ax.imshow(img, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Gradient-based deconvolution (Poisson NLL + TV)")
    fig.tight_layout()
    fig.savefig("deconvolution.png", dpi=130)
    print("  saved deconvolution.png")


if __name__ == "__main__":
    main()
