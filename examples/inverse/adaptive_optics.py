"""Adaptive-optics correction by maximizing image sharpness.

A microscope has an unknown static aberration.  We have a 'deformable mirror'
that can add a Zernike phase to the pupil.  Without ever measuring the
aberration directly, we drive the mirror to *maximize the focal-plane peak
intensity* (a sharpness / Strehl metric) - sensorless adaptive optics, done by
gradient ascent through microsim's differentiable PSF.

At the optimum the applied correction is the negative of the system aberration,
restoring a diffraction-limited focus.

Run:  python examples/inverse/adaptive_optics.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from microsim.inverse import fit, zernike_coeffs
from microsim.optics import vectorial_psf

NA, NI, WVL, DXY, NX = 1.2, 1.515, 0.55, 0.05, 64
ANSI = (5, 6, 7, 8, 9, 12)
# the (unknown to the optimizer) static system aberration, um RMS wavefront
SYSTEM_ABERRATION = jnp.array([0.06, -0.04, 0.05, 0.03, -0.02, 0.05])


def focal_peak(correction: jax.Array) -> jax.Array:
    """Peak focal intensity when the mirror adds `correction` to the system."""
    coeffs = SYSTEM_ABERRATION + correction
    psf = vectorial_psf(
        nz=1,
        nx=NX,
        dxy=DXY,
        dz=0.1,
        wvl=WVL,
        na=NA,
        ni=NI,
        ansi_indices=ANSI,
        coefficients=coeffs,
        normalize="sum",
    )
    return psf[0].max()


def correct(steps: int = 250, lr: float = 3e-3):
    """Find the mirror correction that maximizes focal sharpness."""
    init = zernike_coeffs(ANSI, 0.0)
    return fit(lambda c: -focal_peak(c), init, steps=steps, learning_rate=lr)


def strehl_fraction(correction: jax.Array) -> float:
    aberrated = float(focal_peak(jnp.zeros(len(ANSI))))
    ideal = float(focal_peak(-SYSTEM_ABERRATION))
    corrected = float(focal_peak(correction))
    return (corrected - aberrated) / (ideal - aberrated)


def main() -> None:
    result = correct()
    correction = jnp.asarray(result.params)

    print("Sensorless adaptive-optics correction:")
    print(f"  system aberration : {np.asarray(SYSTEM_ABERRATION)}")
    print(f"  applied correction: {np.round(np.asarray(correction), 4)}")
    print(
        f"  residual |corr+aberr|: "
        f"{np.abs(np.asarray(correction) + np.asarray(SYSTEM_ABERRATION)).max():.4f} um"
    )
    print(f"  Strehl recovered  : {strehl_fraction(correction) * 100:.1f}%")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def psf_img(corr: jax.Array) -> np.ndarray:
        coeffs = SYSTEM_ABERRATION + corr
        psf = vectorial_psf(
            nz=1,
            nx=NX,
            dxy=DXY,
            dz=0.1,
            wvl=WVL,
            na=NA,
            ni=NI,
            ansi_indices=ANSI,
            coefficients=coeffs,
            normalize="sum",
        )
        return np.asarray(psf[0])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    vmax = psf_img(-SYSTEM_ABERRATION).max()
    for ax, img, title in [
        (axs[0], psf_img(jnp.zeros(len(ANSI))), "aberrated PSF"),
        (axs[1], psf_img(correction), "corrected PSF"),
        (axs[2], psf_img(-SYSTEM_ABERRATION), "ideal PSF"),
    ]:
        ax.imshow(img, cmap="magma", vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Sensorless adaptive optics by gradient ascent on sharpness")
    fig.tight_layout()
    fig.savefig("adaptive_optics.png", dpi=130)
    print("  saved adaptive_optics.png")


if __name__ == "__main__":
    main()
