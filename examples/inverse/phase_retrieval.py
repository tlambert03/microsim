"""Zernike phase retrieval by gradient descent.

Given a measured through-focus PSF z-stack of a point source, recover the
pupil aberration (a set of Zernike coefficients) that produced it.

This is the canonical demonstration of microsim's differentiable forward model:
the PSF is built from a Zernike pupil in JAX, so we can take the gradient of an
image-domain loss with respect to the Zernike coefficients and minimize it with
optax - instead of an iterative Gerchberg-Saxton scheme.  Because the forward
model is exact, gradient descent recovers the coefficients essentially perfectly,
even from Poisson-noisy data.

Run:  python examples/inverse/phase_retrieval.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from microsim.inverse import fit, normalized_mse, zernike_coeffs
from microsim.optics import vectorial_psf

# --- imaging configuration (microns) ---
NA, NI, WVL = 1.3, 1.515, 0.55
DXY, DZ, NZ, NX = 0.05, 0.15, 15, 64

# Zernike modes to retrieve (ANSI/OSA).  We deliberately omit piston/tip/tilt/
# defocus (0-4): those are degenerate with the source position and focus.
#   5: vertical astigmatism   6: vertical trefoil   7,8: coma   9: oblique trefoil
#   12: primary spherical
ANSI = (5, 6, 7, 8, 9, 12)


def psf_stack(coeffs: jax.Array) -> jax.Array:
    """Vectorial PSF z-stack for a given Zernike-coefficient vector (um RMS)."""
    return vectorial_psf(
        nz=NZ,
        nx=NX,
        dxy=DXY,
        dz=DZ,
        wvl=WVL,
        na=NA,
        ni=NI,
        ansi_indices=ANSI,
        coefficients=coeffs,
        normalize="sum",
    )


def simulate_measurement(
    true_coeffs: jax.Array, photons: float = 3000.0, seed: int = 0
) -> jax.Array:
    """A 'measured' PSF stack: forward model + Poisson photon noise."""
    psf = psf_stack(true_coeffs)
    scaled = psf / psf.max() * photons
    noisy = jax.random.poisson(jax.random.PRNGKey(seed), scaled)
    return noisy.astype(float) / photons * psf.max()


def retrieve(measured: jax.Array, steps: int = 400, lr: float = 3e-3):
    """Recover Zernike coefficients from a measured PSF stack."""

    def loss_fn(coeffs: jax.Array) -> jax.Array:
        return normalized_mse(psf_stack(coeffs), measured)

    init = zernike_coeffs(ANSI, 0.0)
    return fit(loss_fn, init, steps=steps, learning_rate=lr)


def main() -> None:
    true_coeffs = jnp.array([0.05, -0.03, 0.04, 0.02, 0.0, 0.04])  # um RMS wavefront
    measured = simulate_measurement(true_coeffs)

    result = retrieve(measured)
    recovered = np.asarray(result.params)

    print("Zernike phase retrieval (um RMS wavefront):")
    print(f"  ANSI indices : {ANSI}")
    print(f"  true         : {np.asarray(true_coeffs)}")
    print(f"  recovered    : {np.round(recovered, 4)}")
    print(
        f"  max error    : {np.abs(recovered - np.asarray(true_coeffs)).max():.4f} um"
    )
    print(f"  final loss   : {result.losses[-1]:.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    est = np.asarray(psf_stack(jnp.asarray(recovered)))
    meas = np.asarray(measured)
    mid = NZ // 2
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    for ax, img, title in [
        (axs[0, 0], meas[mid], "measured (focus)"),
        (axs[0, 1], est[mid], "recovered (focus)"),
        (axs[0, 2], meas[mid] - est[mid], "residual (focus)"),
        (axs[1, 0], meas[:, NX // 2], "measured (xz)"),
        (axs[1, 1], est[:, NX // 2], "recovered (xz)"),
        (axs[1, 2], meas[:, NX // 2] - est[:, NX // 2], "residual (xz)"),
    ]:
        ax.imshow(img, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Gradient-based Zernike phase retrieval")
    fig.tight_layout()
    fig.savefig("phase_retrieval.png", dpi=130)
    print("  saved phase_retrieval.png")


if __name__ == "__main__":
    main()
