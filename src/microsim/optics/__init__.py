"""Differentiable optics: vectorial Zernike-pupil PSF (chromatix-backed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ._psf import centered_z, vectorial_psf

if TYPE_CHECKING:
    import jax

    from microsim.schema.aberration import ZernikeAberration
    from microsim.schema.lens import ObjectiveLens

__all__ = ["centered_z", "make_psf", "vectorial_psf"]


def make_psf(
    *,
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    objective: ObjectiveLens,
    ex_wvl_nm: float | None = None,
    em_wvl_nm: float | None = None,
    pinhole_au: float | None = None,
    aberration: ZernikeAberration | None = None,
    pz: float = 0.0,
    normalize: Literal["sum", "max"] | None = "sum",
) -> jax.Array:
    """Build a (widefield or confocal) PSF for an `ObjectiveLens`.

    Wavelengths are in nm; spatial scales in microns.  `aberration` adds a
    Zernike pupil phase (the differentiable knob).  A `pinhole_au` produces a
    confocal PSF = excitation·(emission ⊛ pinhole).
    """
    em_um = em_wvl_nm or ex_wvl_nm
    ex_um = ex_wvl_nm or em_wvl_nm
    if em_um is None:
        raise ValueError("Either excitation or emission wavelength is required.")
    em_um = em_um * 1e-3
    ex_um = ex_um * 1e-3

    ansi = aberration.ansi_indices if aberration else ()
    coeffs = aberration.as_array() if aberration else None

    def _psf(wvl_um: float) -> jax.Array:
        return vectorial_psf(
            nz=nz,
            nx=nx,
            dxy=dx,
            dz=dz,
            wvl=wvl_um,
            na=objective.numerical_aperture,
            ni=objective.immersion_medium_ri,
            pz=pz,
            ansi_indices=ansi,
            coefficients=coeffs,
            normalize=normalize,
        )

    if pinhole_au is None:
        return _psf(em_um)
    return _confocal_psf(
        _psf(ex_um),
        _psf(em_um),
        pinhole_au=pinhole_au,
        em_wvl_um=em_um,
        na=objective.numerical_aperture,
        dxy=dx,
        normalize=normalize,
    )


def _confocal_psf(
    ex_psf: jax.Array,
    em_psf: jax.Array,
    *,
    pinhole_au: float,
    em_wvl_um: float,
    na: float,
    dxy: float,
    normalize: Literal["sum", "max"] | None,
) -> jax.Array:
    import jax
    import jax.numpy as jnp
    from jax.scipy.signal import fftconvolve

    from ._psf import _normalize

    nxy = ex_psf.shape[-1]
    pinhole_px = (pinhole_au * 0.61 * em_wvl_um / na) / dxy
    x = jnp.arange(nxy) - nxy // 2
    xx, yy = jnp.meshgrid(x, x)
    pinhole = (jnp.hypot(xx, yy) <= pinhole_px).astype(float)

    # effective emission PSF: each plane blurred by the pinhole
    eff_em = jax.vmap(lambda p: fftconvolve(p, pinhole, mode="same"))(em_psf)
    return _normalize(ex_psf * eff_em, normalize)
