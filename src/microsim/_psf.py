"""Point spread functions: implementation (see `microsim.psf` for the public API)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import tqdm

from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveKwargs, ObjectiveLens
from microsim.schema.settings import Settings
from microsim.util import microsim_cache

from ._data_array import ArrayProtocol
from ._logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
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
    return 8.0 * np.pi / 3.0 * simpson_integral * (step / ud) ** 2  # type: ignore[no-any-return]


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
    return out.get() if hasattr(out, "get") else out


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


@dataclass(frozen=True)
class PSFGrid:
    """Shape and voxel size (in um) of a 3D PSF array.

    Sizes are rounded up to odd numbers, so that the PSF is centered on a voxel (an
    even-sized PSF shifts the image by a fraction of a pixel when convolved).
    """

    nz: int
    nx: int
    dz: float
    dx: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "nz", int(self.nz) | 1)
        object.__setattr__(self, "nx", int(self.nx) | 1)


def airy_radius_um(wvl_um: float, na: float) -> float:
    """Radius of the Airy disk (to its first minimum), in um."""
    return 0.61 * wvl_um / na


def apply_saturation(spot: Any, saturation: float) -> Any:
    """Apply excitation saturation to a relative excitation intensity (1 at focus).

    `saturation` is the fluorophore's saturation at the reference (focal) intensity
    (see `ChannelFluorescence.saturation`).  The returned array is the emission at
    each position, relative to the *unsaturated* emission at the focus.
    """
    if not saturation:
        return spot
    return spot / (1 + saturation * spot)


def _excitation_spot(
    nz: int,
    wvl_um: float,
    dz: float,
    nx: int,
    dxy: float,
    objective: ObjectiveLens,
    xp: NumpyAPI,
    pz: float = 0,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    sf: int = 3,
) -> npt.NDArray:
    """Relative excitation intensity of a focused laser spot (1 at the focus)."""
    return vectorial_psf_centered(
        nz=nz,
        wvl=wvl_um,
        dz=dz,
        pz=pz,
        nx=nx,
        ny=ny,
        pos=pos,
        dxy=dxy,
        objective=objective,
        xp=xp,
        sf=sf,
        normalize="max",
    )


def _detection_psf(
    nz: int,
    wvl_um: float,
    dz: float,
    nx: int,
    dxy: float,
    objective: ObjectiveLens,
    xp: NumpyAPI,
    pinhole_radius_um: float | None = None,
    pz: float = 0,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    sf: int = 3,
) -> npt.NDArray:
    """Fraction of the light collected from a point source that reaches the detector.

    Without a pinhole, each z-plane is the image of the point source on a camera
    (see `_norm_to_collected_light`).  With a pinhole (radius in sample space), each
    voxel is the fraction of the collected light passing the pinhole, for a point
    source at that position relative to the focus.
    """
    psf = vectorial_psf_centered(
        nz=nz,
        wvl=wvl_um,
        dz=dz,
        pz=pz,
        nx=nx,
        ny=ny,
        pos=pos,
        dxy=dxy,
        objective=objective,
        xp=xp,
        sf=sf,
        normalize=False,
    )
    psf = _norm_to_collected_light(psf, wvl_um, dxy, objective, sf, xp)
    if pinhole_radius_um is None:
        return psf

    # light through the pinhole: the image of the point source, summed over the
    # pinhole area (i.e. convolved with the pinhole mask)
    pinhole = xp.asarray(_pinhole_mask(psf.shape[-1], pinhole_radius_um, dxy))
    out = xp.empty_like(psf)
    for i in tqdm.trange(len(psf), desc="convolving em_psf with pinhole..."):
        plane = xp.fftconvolve(xp.asarray(psf[i]), pinhole, mode="same")
        out = xp._array_assign(out, i, plane)
    return out  # type: ignore [no-any-return]


# lateral radius (in Airy units) of the in-focus plane used to normalize emission
# PSFs (see `_norm_to_collected_light`).  Light beyond it is ~0.17 / 64 = ~0.3%.
_NORM_RADIUS_AU = 64


def _norm_to_collected_light(
    psf: np.ndarray,
    wvl_um: float,
    dxy: float,
    objective: ObjectiveLens,
    sf: int,
    xp: NumpyAPI,
) -> np.ndarray:
    """Scale an emission PSF to the fraction of collected light on each pixel.

    Each z-plane is the image of a point source at that defocus.  The vectorial model
    conserves energy across planes, so every plane holds the same total, and dividing
    by it makes each plane sum to ~1: a camera focused at any depth receives all the
    light collected from the point source.  (Collection efficiency itself is applied
    to the detected fraction of emitted light, in `ChannelFluorescence.create`.)

    The total is the sum of the in-focus plane over at least `_NORM_RADIUS_AU`, rather
    than over `psf` itself: if `psf` is laterally truncated, its own planes miss some
    light, and normalizing by them would brighten the PSF to compensate.
    """
    airy_px = 0.61 * wvl_um / objective.numerical_aperture / dxy
    nx = max(psf.shape[-1], int(2 * _NORM_RADIUS_AU * airy_px) | 1)
    return psf / _in_focus_sum(nx, dxy, wvl_um, objective, sf, xp)


@cache
def _in_focus_sum(
    nx: int, dxy: float, wvl_um: float, objective: ObjectiveLens, sf: int, xp: NumpyAPI
) -> float:
    """Sum of the (unnormalized) in-focus plane of an emission PSF of width `nx`."""
    plane = vectorial_psf_centered(
        nz=1,
        nx=nx,
        dxy=dxy,
        wvl=wvl_um,
        objective=objective,
        sf=sf,
        normalize=False,
        xp=xp,
    )
    return float(xp.sum(plane))


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


# minimum pinhole radius, in subpixels, used to compute the open area of pixels at
# the pinhole edge (area error < 0.5%)
_PINHOLE_RADIUS_SUBPX = 64


def _pinhole_mask(nxy: int, radius_um: float, dxy: float) -> npt.NDArray:
    """Create a 2D circular pinhole mask of radius `radius_um` (in sample space).

    Each pixel is the fraction of its area inside the pinhole, so the total open
    area matches the pinhole's (a pinhole only a few pixels across would otherwise
    snap to whole pixels, e.g. 9 px instead of 11.3 px for a radius of 1.9 px).
    """
    radius_px = radius_um / dxy
    mask = np.zeros((nxy, nxy))
    center = nxy // 2
    if radius_px < 0.5:  # the pinhole lies within the central pixel
        mask[center, center] = np.pi * radius_px**2
        return mask

    # only pixels near the pinhole can be (partly) open
    half = min(int(np.ceil(radius_px)) + 1, (nxy - 1) // 2)
    x = np.arange(-half, half + 1)
    ss = max(4, int(np.ceil(_PINHOLE_RADIUS_SUBPX / radius_px)))
    sub = (x[:, None] + (np.arange(ss) + 0.5) / ss - 0.5).ravel()
    inside = np.hypot(sub[:, None], sub[None, :]) <= radius_px
    local = inside.reshape(x.size, ss, x.size, ss).mean(axis=(1, 3))
    mask[center - half : center + half + 1, center - half : center + half + 1] = local
    return mask


@lru_cache(maxsize=Settings().cache.in_mem_size.psf)
def excitation_spot(
    grid: PSFGrid, wvl_nm: float, objective: ObjectiveLens, xp: NumpyAPI
) -> ArrayProtocol:
    """Relative excitation intensity of a focused laser spot (1 at the focus).

    The result is cached (in memory, and on disk), so it must not be modified.
    """
    return _disk_cached(
        ("spot", wvl_nm),
        grid,
        objective,
        xp,
        lambda: _excitation_spot(
            grid.nz, wvl_nm * 1e-3, grid.dz, grid.nx, grid.dx, objective, xp
        ),
    )


@lru_cache(maxsize=Settings().cache.in_mem_size.psf)
def detection_psf(
    grid: PSFGrid,
    wvl_nm: float,
    objective: ObjectiveLens,
    xp: NumpyAPI,
    pinhole_radius_um: float | None = None,
) -> ArrayProtocol:
    """Fraction of the light collected from a point source that reaches the detector.

    For a camera (no pinhole), each z-plane is the image of a point source at that
    distance from the focal plane, as a fraction of the light collected from it.
    With a pinhole (radius in um, in sample space), each voxel is the fraction of the
    collected light that passes the pinhole, for a point source at that position.

    The result is cached (in memory, and on disk), so it must not be modified.
    """
    name: tuple[str | float, ...] = ("detection", wvl_nm)
    if pinhole_radius_um is not None:
        name = (*name, "pinhole", pinhole_radius_um)
    return _disk_cached(
        name,
        grid,
        objective,
        xp,
        lambda: _detection_psf(
            grid.nz,
            wvl_nm * 1e-3,
            grid.dz,
            grid.nx,
            grid.dx,
            objective,
            xp,
            pinhole_radius_um=pinhole_radius_um,
        ),
    )


def _disk_cached(
    name: tuple[str | float, ...],
    grid: PSFGrid,
    objective: ObjectiveLens,
    xp: NumpyAPI,
    compute: Callable[[], ArrayProtocol],
) -> ArrayProtocol:
    """Load a PSF array from the disk cache, or compute (and cache) it."""
    use_cache = os.getenv("MICROSIM_CACHE", "").lower() not in {"0", "false", "no", "n"}
    path = _psf_cache_path(name, grid, objective)
    if use_cache and path.exists():
        logger.info(f"Found cached PSF {path.stem}")
        # the file may have been written by a simulation of another precision
        return _read_only(xp.asarray(np.load(path), dtype=xp.float_dtype))

    logger.info(f"Creating new PSF {path.stem}")
    psf = compute()
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, psf)
    return _read_only(xp.asarray(psf))


def _read_only(psf: ArrayProtocol) -> ArrayProtocol:
    """Protect a cached (shared) numpy array from in-place modification."""
    if isinstance(psf, np.ndarray):
        psf.setflags(write=False)
    return psf


def _psf_cache_path(
    name: tuple[str | float, ...], grid: PSFGrid, objective: ObjectiveLens
) -> Path:
    """Return the cache location for a PSF array."""
    # "v4": PSFs are cached as separate excitation spots and detection PSFs
    # "v5": anti-aliased pinhole mask
    # e.g. 520 and 520.0 nm share a file
    name = tuple(float(x) if isinstance(x, int | float) else x for x in name)
    parts = [*name, grid.nz, grid.nx, grid.dz, grid.dx, "v5"]
    stem = "_".join(str(x).replace(".", "-") for x in parts)
    return (microsim_cache("psf") / objective.cache_key() / stem).with_suffix(".npy")
