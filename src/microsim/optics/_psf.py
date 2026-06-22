"""Differentiable vectorial PSF model, backed by chromatix.

The point spread function is built from a complex pupil:

    pupil(k) = aperture(k) · exp(i · 2π/λ · Σ cᵢ Zᵢ(k))

and propagated to the focal plane with chromatix's high-NA (Debye-Wolf)
vectorial focusing.  Everything here is pure JAX, so the resulting PSF is
differentiable with respect to the Zernike `coefficients` (and, in principle,
the optical parameters) - which is what makes gradient-based phase retrieval
and PSF engineering possible.

Units are microns throughout.

Zernike coefficients are wavefront **optical path differences in microns**,
expanded on the ANSI/OSA-indexed, RMS-normalized Zernike basis.  Because they
are an OPD (not a phase), the same coefficients describe the aberration at every
emission wavelength - chromatix turns them into a per-wavelength phase via
`2π·OPD/λ`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from chromatix.functional import high_na_ff_lens, objective_point_source, phase_change
from chromatix.utils import zernike_aberrations

if TYPE_CHECKING:
    from collections.abc import Sequence

# focal length (um). Cancels out of the PSF in physical units (the aperture
# sampling scales with it while the output pixel size is fixed), so the exact
# value is arbitrary; it only sets the internal coordinate scale.
_DEFAULT_F = 3000.0

# transverse polarizations (z, y, x order) averaged to model unpolarized light
_UNPOL = (jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0]))


def centered_z(nz: int, dz: float, pz: float = 0.0) -> jax.Array:
    """`nz` focal-plane offsets (um) centered on `pz`, spaced by `dz`."""
    return (jnp.arange(nz) - (nz - 1) / 2) * dz + pz


def _one_pol_psf(
    *,
    shape: tuple[int, int],
    dx_ap: float,
    wvl: float,
    z: jax.Array,
    f: float,
    ni: float,
    na: float,
    phase: jax.Array | None,
    amplitude: jax.Array,
    out_nx: int,
    dxy: float,
) -> jax.Array:
    field = objective_point_source(
        shape, dx_ap, wvl, z, f, ni, na, scalar=False, amplitude=amplitude
    )
    if phase is not None:
        field = phase_change(field, phase)
    field = high_na_ff_lens(field, f, ni, na, (out_nx, out_nx), dxy)
    return field.intensity.squeeze()  # type: ignore[no-any-return]


def vectorial_psf(
    *,
    nz: int,
    nx: int,
    dxy: float,
    dz: float,
    wvl: float,
    na: float,
    ni: float = 1.515,
    pz: float = 0.0,
    ansi_indices: Sequence[int] = (),
    coefficients: jax.Array | Sequence[float] | None = None,
    pupil_samples: int | None = None,
    polarization: tuple[float, float, float] | None = None,
    f: float = _DEFAULT_F,
    normalize: Literal["sum", "max"] | None = "sum",
) -> jax.Array:
    """Vectorial Zernike-pupil PSF, shape `(nz, nx, nx)`, real intensity.

    Parameters
    ----------
    nz, nx : int
        Output z and lateral pixel counts.
    dxy, dz : float
        Lateral and axial pixel size (um).
    wvl : float
        Wavelength (um).
    na : float
        Numerical aperture.
    ni : float
        Immersion medium refractive index.
    pz : float
        Axial position of the point source relative to focus (um).
    ansi_indices : Sequence[int]
        ANSI/OSA Zernike indices for the aberration.
    coefficients : array-like or None
        Wavefront OPD (um) for each index in `ansi_indices`.  This is the
        differentiable parameter for phase retrieval / PSF engineering.
    pupil_samples : int or None
        Pupil-plane grid size (defaults to a sensible oversampling of `nx`).
    polarization : tuple or None
        Input polarization (z, y, x).  `None` models unpolarized light (average
        of two orthogonal transverse polarizations).
    f : float
        Objective focal length (um).  Cancels in physical units; only sets the
        internal coordinate scale.
    normalize : {"sum", "max"} or None
        PSF normalization.
    """
    pupil_n = pupil_samples or max(128, _next_even(nx))
    # aperture sampling so the circular pupil (diameter 2·f·NA/ni) fills the grid
    dx_ap = 2.0 * (f * na / ni) / pupil_n
    z = centered_z(nz, dz, pz)

    phase: jax.Array | None = None
    if coefficients is not None and len(ansi_indices):
        phase = zernike_aberrations(
            (pupil_n, pupil_n),
            dx_ap,
            wvl,
            ni,
            f,
            na,
            ansi_indices,
            jnp.asarray(coefficients),
            normalize=True,
        )

    pols = (jnp.asarray(polarization),) if polarization is not None else _UNPOL
    psf = jnp.zeros((nz, nx, nx))
    for amp in pols:
        psf = psf + _one_pol_psf(
            shape=(pupil_n, pupil_n),
            dx_ap=dx_ap,
            wvl=wvl,
            z=z,
            f=f,
            ni=ni,
            na=na,
            phase=phase,
            amplitude=amp,
            out_nx=nx,
            dxy=dxy,
        )

    return _normalize(psf, normalize)


def _normalize(psf: jax.Array, mode: Literal["sum", "max"] | None) -> jax.Array:
    if mode == "sum":
        return psf / psf.sum()
    if mode == "max":
        return psf / psf.max()
    return psf


def _next_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1
