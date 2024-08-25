from __future__ import annotations

import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import tqdm

from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveKwargs, ObjectiveLens
from microsim.util import microsim_cache

from ._data_array import ArrayProtocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from microsim._data_array import ArrayProtocol


def simpson(
    objective: ObjectiveLens,
    theta: np.ndarray,
    constJ: np.ndarray,
    zv: np.ndarray,
    ci: float,
    zp: float,
    wave_num: float,
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    xp = NumpyAPI.create(xp)

    # L_theta calculation
    sintheta = xp.sin(theta)
    costheta = xp.cos(theta)
    sqrtcostheta = xp.sqrt(costheta).astype("complex")
    ni2sin2theta = objective.ni**2 * sintheta**2
    nsroot = xp.sqrt(objective.ns**2 - ni2sin2theta)
    ngroot = xp.sqrt(objective.ng**2 - ni2sin2theta)
    _z = xp.asarray(zv if np.isscalar(zv) else zv[:, xp.newaxis, xp.newaxis])
    L0 = (
        objective.ni * (ci - _z) * costheta
        + zp * nsroot
        + objective.tg * ngroot
        - objective.tg0 * xp.sqrt(objective.ng0**2 - ni2sin2theta)
        - objective.ti0 * xp.sqrt(objective.ni0**2 - ni2sin2theta)
    )
    expW = xp.exp(1j * wave_num * L0)

    simp = xp._simp_like(theta)

    ts1ts2 = (4.0 * objective.ni * costheta * ngroot).astype("complex")
    tp1tp2 = ts1ts2.copy()
    tp1tp2 /= (objective.ng * costheta + objective.ni / objective.ng * ngroot) * (
        objective.ns / objective.ng * ngroot + objective.ng / objective.ns * nsroot
    )
    ts1ts2 /= (objective.ni * costheta + ngroot) * (ngroot + nsroot)

    # 2.0 factor: Simpson's rule
    bessel_0 = simp * xp.j0(constJ[:, xp.newaxis] * sintheta) * sintheta * sqrtcostheta
    bessel_1 = simp * xp.j1(constJ[:, xp.newaxis] * sintheta) * sintheta * sqrtcostheta

    with np.errstate(invalid="ignore"):
        bessel_2 = 2.0 * bessel_1 / (constJ[:, xp.newaxis] * sintheta) - bessel_0

    bessel_2 = xp._array_assign(bessel_2, constJ == 0.0, 0)

    bessel_0 *= ts1ts2 + tp1tp2 / objective.ns * nsroot
    bessel_1 *= tp1tp2 * objective.ni / objective.ns * sintheta
    bessel_2 *= ts1ts2 - tp1tp2 / objective.ns * nsroot

    sum_I0 = xp.abs((expW * bessel_0).sum(-1))
    sum_I1 = xp.abs((expW * bessel_1).sum(-1))
    sum_I2 = xp.abs((expW * bessel_2).sum(-1))

    return xp.real(sum_I0**2 + 2.0 * sum_I1**2 + sum_I2**2)  # type: ignore


def _cast_objective(objective: ObjectiveKwargs | ObjectiveLens | None) -> ObjectiveLens:
    if isinstance(objective, ObjectiveLens):
        return objective
    if objective is None or isinstance(objective, dict):
        return ObjectiveLens.model_validate(objective or {})
    raise TypeError(f"Expected ObjectiveLens, got {type(objective)}")


@cache
def vectorial_rz(
    zv: Sequence[float],
    nx: int = 51,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.04,
    wvl: float = 0.6,
    objective: ObjectiveKwargs | ObjectiveLens | None = None,
    sf: int = 3,
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    xp = NumpyAPI.create(xp)
    p = _cast_objective(objective)
    wave_num = 2 * np.pi / (wvl * 1e-6)

    xpos, ypos, zpos = pos

    # nz_ = len(z)
    xystep_ = dxy * 1e-6
    xymax = (nx * sf - 1) // 2

    # position in pixels
    xpos *= sf / xystep_
    ypos *= sf / xystep_
    rn = 1 + int(xp.sqrt(xpos * xpos + ypos * ypos))
    rmax = int(xp.ceil(np.sqrt(2.0) * xymax) + rn + 1)  # +1 for interpolation, dx, dy
    rvec = xp.arange(rmax) * xystep_ / sf
    constJ = wave_num * rvec * p.ni

    # CALCULATE
    # constant component of OPD
    ci = zpos * (1 - p.ni / p.ns) + p.ni * (p.tg0 / p.ng0 + p.ti0 / p.ni0 - p.tg / p.ng)

    nSamples = 4 * int(1.0 + p.half_angle * xp.max(constJ) / np.pi)
    nSamples = np.maximum(nSamples, 60)
    ud = 3.0 * sf

    step = p.half_angle / nSamples
    theta = xp.arange(1, nSamples + 1) * step
    simpson_integral = simpson(
        p, theta, constJ, xp.asarray(zv), ci, zpos, wave_num, xp=xp
    )
    return 8.0 * np.pi / 3.0 * simpson_integral * (step / ud) ** 2


def radius_map(
    shape: Sequence[int], off: Sequence[int] | None = None, xp: NumpyAPI | None = None
) -> npt.NDArray:
    xp = NumpyAPI.create(xp)
    if off is not None:
        offy, offx = off
    else:
        off = (0, 0)
    ny, nx = shape
    yi, xi = xp.mgrid[:ny, :nx]
    yi = yi - (ny - 1) / 2 - offy
    xi = xi - (nx - 1) / 2 - offx
    return xp.hypot(yi, xi)  # type: ignore


def rz_to_xyz(
    rz: npt.NDArray,
    xyshape: Sequence[int],
    sf: int = 3,
    off: Sequence[int] | None = None,
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    """Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF."""
    xp = NumpyAPI.create(xp)

    # Create XY grid of radius values.
    rmap = radius_map(xyshape, off, xp=xp) * sf
    nz = rz.shape[0]

    out = xp.asarray(
        [
            xp.map_coordinates(
                rz, xp.asarray([xp.ones(rmap.size) * z, rmap.ravel()]), order=1
            ).reshape(xyshape)
            for z in range(nz)
        ]
    )
    return out.get() if hasattr(out, "get") else out  # type: ignore


# def rz_to_xyz(rz, xyshape, sf=3, off=None):
#     """Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF."""
#     # Create XY grid of radius values.
#     rmap = radius_map(xyshape, off) * sf
#     ny, nx = xyshape
#     nz, nr = rz.shape
#     ZZ, RR = xp.meshgrid(xp.arange(nz, dtype="float64"), rmap.ravel())
#     o = map_coordinates(rz, xp.array([ZZ.ravel(), RR.ravel()]), order=1)
#     return o.reshape((nx, ny, nz)).T


def vectorial_psf(
    zv: Sequence[float],
    nx: int = 31,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.05,
    wvl: float = 0.6,
    objective: ObjectiveKwargs | ObjectiveLens | None = None,
    sf: int = 3,
    normalize: bool | Literal["sum", "max"] = "sum",
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    xp = NumpyAPI.create(xp)
    zv = tuple(np.asarray(zv) * 1e-6)  # convert to meters
    ny = ny or nx
    rz = vectorial_rz(
        zv, np.maximum(ny, nx), pos, dxy, wvl, objective=objective, sf=sf, xp=xp
    ).astype(xp.float_dtype)

    offsets = xp.asarray(pos[:2]) / (dxy * 1e-6)
    _psf = rz_to_xyz(rz, (ny, nx), sf, off=offsets, xp=xp)  # type: ignore [arg-type]
    _psf = _norm_psf(_psf, normalize, xp)
    return _psf


def _centered_zv(nz: int, dz: float, pz: float = 0) -> tuple[float, ...]:
    lim = (nz - 1) * dz / 2
    return tuple(np.linspace(-lim + pz, lim + pz, nz))


def vectorial_psf_centered(
    nz: int,
    dz: float = 0.05,
    pz: float = 0,
    nx: int = 31,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.05,
    wvl: float = 0.6,
    objective: ObjectiveKwargs | ObjectiveLens | None = None,
    sf: int = 3,
    normalize: bool | Literal["sum", "max"] = "sum",
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    """Compute a vectorial model of the microscope point spread function.

    The point source is always in the center of the output volume.
    """
    return vectorial_psf(
        zv=_centered_zv(nz, dz, pz),
        nx=nx,
        ny=ny,
        pos=pos,
        dxy=dxy,
        wvl=wvl,
        objective=objective,
        sf=sf,
        normalize=normalize,
        xp=xp,
    )


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
    xp: NumpyAPI | None = None,
) -> np.ndarray:
    """Create a confocal PSF.

    This function creates a confocal PSF by multiplying the excitation PSF with
    the emission PSF convolved with a pinhole mask.

    All extra keyword arguments are passed to `vectorial_psf_centered`.
    """
    xp = NumpyAPI.create(xp)

    objective = _cast_objective(objective)
    ex_psf = vectorial_psf_centered(
        nz=nz,
        wvl=ex_wvl_um,
        dz=dz,
        pz=pz,
        nx=nx,
        ny=ny,
        pos=pos,
        dxy=dxy,
        objective=objective,
        xp=xp,
        sf=sf,
        normalize=normalize,
    )
    em_psf = vectorial_psf_centered(
        nz=nz,
        wvl=em_wvl_um,
        dz=dz,
        pz=pz,
        nx=nx,
        ny=ny,
        pos=pos,
        dxy=dxy,
        objective=objective,
        xp=xp,
        sf=sf,
        normalize=normalize,
    )

    # The effective emission PSF is the regular emission PSF convolved with the
    # pinhole mask. The pinhole mask is a disk with diameter equal to the pinhole
    # size in AU, converted to pixels.
    pinhole = _pinhole_mask(
        nxy=ex_psf.shape[-1],
        pinhole_au=pinhole_au,
        wvl=em_wvl_um,
        na=objective.numerical_aperture,
        dxy=dxy,
        xp=xp,
    )
    pinhole = xp.asarray(pinhole)

    eff_em_psf = xp.empty_like(em_psf)
    for i in tqdm.trange(len(em_psf), desc="convolving em_psf with pinhole..."):
        plane = xp.fftconvolve(xp.asarray(em_psf[i]), pinhole, mode="same")
        eff_em_psf = xp._array_assign(eff_em_psf, i, plane)

    # The final PSF is the excitation PSF multiplied by the effective emission PSF.
    out = xp.asarray(ex_psf) * eff_em_psf
    out = _norm_psf(out, normalize, xp)
    return out  # type: ignore [no-any-return]


def _norm_psf(
    psf: np.ndarray, normalize: bool | Literal["sum", "max"], xp: NumpyAPI
) -> np.ndarray:
    if normalize:
        if isinstance(normalize, str):
            if normalize == "max":
                psf /= xp.max(psf)
            if normalize == "sum":
                psf /= xp.sum(psf)
        else:
            psf /= xp.sum(psf)
    return psf


def _pinhole_mask(
    nxy: int,
    pinhole_au: float,
    wvl: float,
    na: float,
    dxy: float,
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    """Create a 2D circular pinhole mask of specified `pinhole_au`."""
    xp = NumpyAPI.create(xp)

    pinhole_size = pinhole_au * 0.61 * wvl / na
    pinhole_px = pinhole_size / dxy

    x = xp.arange(nxy) - nxy // 2
    xx, yy = xp.meshgrid(x, x)
    r = xp.sqrt(xx**2 + yy**2)
    return (r <= pinhole_px).astype(int)  # type: ignore


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
    xp: NumpyAPI | None = None,
) -> ArrayProtocol:
    if ex_wvl_nm is None:
        if em_wvl_nm is None:
            raise ValueError(
                "Either excitation or emission must be specified to generate a PSF."
            )
        ex_wvl_nm = em_wvl_nm
    if em_wvl_nm is None:
        em_wvl_nm = ex_wvl_nm

    return cached_psf(
        nz=nz,
        nx=nx,
        dx=dx,
        dz=dz,
        ex_wvl_um=ex_wvl_nm * 1e-3,  # nm to um
        em_wvl_um=em_wvl_nm * 1e-3,  # nm to um
        objective=_cast_objective(objective),
        pinhole_au=pinhole_au,
        max_au_relative=max_au_relative,
        xp=NumpyAPI.create(xp),
    )


# variant of make_psf that only accepts hashable arguments
@cache
def cached_psf(
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    ex_wvl_um: float,
    em_wvl_um: float,
    objective: ObjectiveLens,
    pinhole_au: float | None,
    max_au_relative: float | None,
    xp: NumpyAPI,
) -> ArrayProtocol:
    # now restrict nx to no more than max_au_relative
    if max_au_relative is not None:
        airy_radius = 0.61 * ex_wvl_um / objective.numerical_aperture
        n_pix_per_airy_radius = airy_radius / dx
        max_nx = int(n_pix_per_airy_radius * max_au_relative * 2)
        nx = min(nx, max_nx)
        # if even make odd
        if nx % 2 == 0:
            nx += 1

    use_cache = os.getenv("MICROSIM_CACHE", "").lower() not in {"0", "false", "no", "n"}
    if use_cache:
        cache_path = _psf_cache_path(
            nz, nx, dz, dx, em_wvl_um, pinhole_au, ex_wvl_um, objective
        )
        if cache_path.exists():
            logging.info("Using cached PSF: %s", cache_path)
            return xp.asarray(np.load(cache_path))

    if pinhole_au is None:
        psf = vectorial_psf_centered(
            wvl=em_wvl_um,
            nz=nz + 1,
            nx=nx + 1,
            dz=dz,
            dxy=dx,
            objective=objective,
            xp=xp,
            normalize="sum",
        )
    else:
        psf = make_confocal_psf(
            nz=nz,
            ex_wvl_um=ex_wvl_um,
            em_wvl_um=em_wvl_um,
            pinhole_au=pinhole_au,
            nx=nx,
            dz=dz,
            dxy=dx,
            objective=objective,
            xp=xp,
            normalize="sum",
        )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, psf)
    return xp.asarray(psf)


def _psf_cache_path(
    nz: int,
    nx: int,
    dz: float,
    dx: float,
    em_wvl_um: float,
    pinhole_au: float | None,
    ex_wvl_um: float,
    objective: ObjectiveLens,
) -> Path:
    """Return the cache location for these PSF parameters."""
    cache_key = [nz, nx, dz, dx, em_wvl_um]
    if pinhole_au is not None:
        cache_key.extend([ex_wvl_um, pinhole_au])
    cache_path = microsim_cache("psf") / objective.cache_key()
    cache_path = cache_path / "_".join([str(x).replace(".", "-") for x in cache_key])
    return cache_path.with_suffix(".npy")
