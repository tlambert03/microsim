"""Point spread functions.

The implementation lives in `microsim._psf`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from microsim import _psf
from microsim._psf import (
    radius_map,
    rz_to_xyz,
    simpson,
    vectorial_psf,
    vectorial_psf_centered,
    vectorial_rz,
)
from microsim.schema.backend import NumpyAPI

if TYPE_CHECKING:
    import numpy as np

    from microsim._data_array import ArrayProtocol
    from microsim.schema.lens import ObjectiveKwargs, ObjectiveLens

__all__ = [
    "make_confocal_psf",
    "make_psf",
    "radius_map",
    "rz_to_xyz",
    "simpson",
    "vectorial_psf",
    "vectorial_psf_centered",
    "vectorial_rz",
]


def make_confocal_psf(
    nz: int,
    ex_wvl_um: float = 0.475,
    em_wvl_um: float = 0.525,
    pinhole_au: float = 1.0,
    dz: float = 0.05,
    pz: float = 0,
    nx: int = 31,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.05,
    objective: ObjectiveKwargs | ObjectiveLens | None = None,
    sf: int = 3,
    normalize: bool | Literal["sum", "max"] = "sum",
    saturation: float = 0,
    xp: NumpyAPI | None = None,
) -> np.ndarray:
    """Create a confocal PSF.

    This function creates a confocal system PSF: the excitation spot times the
    detection PSF (the emission PSF convolved with a pinhole mask).  `pinhole_au` is
    in Airy units at the emission wavelength.

    If `normalize` is False, the result is physically scaled: the relative excitation
    intensity (1 at focus) times the fraction of collected emission that passes the
    pinhole, for a point source at each position.

    `saturation` is the excitation saturation parameter at the focus (`k * tau`: the
    absorption rate times the excited-state lifetime).  If non-zero, the relative
    excitation `r` is replaced by the relative emission of a two-level system,
    `r / (1 + s * r)`.

    All extra keyword arguments are passed to `vectorial_psf_centered`.
    """
    xp = NumpyAPI.create(xp)
    objective = _psf._cast_objective(objective)
    kw: dict[str, Any] = {"nz": nz, "dz": dz, "pz": pz, "nx": nx, "ny": ny}
    kw.update(pos=pos, dxy=dxy, objective=objective, sf=sf, xp=xp)
    spot = _psf.apply_saturation(
        _psf._excitation_spot(wvl_um=ex_wvl_um, **kw), saturation
    )
    pinhole_um = pinhole_au * _psf.airy_radius_um(
        em_wvl_um, objective.numerical_aperture
    )
    det = _psf._detection_psf(wvl_um=em_wvl_um, pinhole_radius_um=pinhole_um, **kw)
    return _psf._norm_psf(xp.asarray(spot) * det, normalize, xp)


def make_psf(
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    objective: ObjectiveLens,
    ex_wvl_nm: float | None = None,
    em_wvl_nm: float | None = None,
    pinhole_au: float | None = None,
    max_au_relative: float | None = None,
    saturation: float = 0,
    xp: NumpyAPI | None = None,
) -> ArrayProtocol:
    """Make a widefield PSF, or (if `pinhole_au` is given) a confocal PSF.

    Results are cached (in memory, and on disk).  `pinhole_au` is
    in Airy units at the emission wavelength, and `max_au_relative` (if given) caps
    the lateral size of the PSF, in Airy units at the excitation wavelength.
    """
    if ex_wvl_nm is None:
        if em_wvl_nm is None:
            raise ValueError(
                "Either excitation or emission must be specified to generate a PSF."
            )
        ex_wvl_nm = em_wvl_nm
    if em_wvl_nm is None:
        em_wvl_nm = ex_wvl_nm
    ex_wvl_nm, em_wvl_nm = float(ex_wvl_nm), float(em_wvl_nm)

    objective = _psf._cast_objective(objective)
    na = objective.numerical_aperture
    if max_au_relative is not None:
        airy_px = _psf.airy_radius_um(ex_wvl_nm * 1e-3, na) / dx
        nx = min(nx, int(airy_px * max_au_relative * 2))
    grid = _psf.PSFGrid(nz=nz, nx=nx, dz=dz, dx=dx)
    xp = NumpyAPI.create(xp)

    if pinhole_au is None:
        # a copy: the cached PSF must not be modified
        return _psf.detection_psf(grid, em_wvl_nm, objective, xp).copy()  # type: ignore [attr-defined, no-any-return]
    pinhole_um = pinhole_au * _psf.airy_radius_um(em_wvl_nm * 1e-3, na)
    det = _psf.detection_psf(
        grid, em_wvl_nm, objective, xp, pinhole_radius_um=pinhole_um
    )
    spot = _psf.apply_saturation(
        _psf.excitation_spot(grid, ex_wvl_nm, objective, xp), saturation
    )
    return spot * det  # type: ignore [no-any-return]
