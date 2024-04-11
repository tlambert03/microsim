from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveKwargs, ObjectiveLens


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
    _z = zv[:, xp.newaxis, xp.newaxis] if zv.ndim else zv
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


def vectorial_rz(
    zv: npt.NDArray,
    nx: int = 51,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.04,
    wvl: float = 0.6,
    objective_params: ObjectiveKwargs | None = None,
    sf: int = 3,
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    xp = NumpyAPI.create(xp)
    p = ObjectiveLens(**(objective_params or {}))

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
    simpson_integral = simpson(p, theta, constJ, zv, ci, zpos, wave_num)
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
    rmap = radius_map(xyshape, off) * sf
    nz = rz.shape[0]
    out = xp.zeros((nz, *xyshape))
    out = []
    for z in range(nz):
        o = xp.map_coordinates(
            rz, xp.asarray([xp.ones(rmap.size) * z, rmap.ravel()]), order=1
        ).reshape(xyshape)
        out.append(o)

    out = xp.asarray(out)
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
    zv: npt.NDArray,
    nx: int = 31,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.05,
    wvl: float = 0.6,
    objective_params: ObjectiveKwargs | None = None,
    sf: int = 3,
    normalize: bool = True,
    xp: NumpyAPI | None = None,
) -> npt.NDArray:
    xp = NumpyAPI.create(xp)
    zv = xp.asarray(zv * 1e-6)  # convert to meters
    ny = ny or nx
    rz = vectorial_rz(
        zv,
        np.maximum(ny, nx),
        pos,
        dxy,
        wvl,
        objective_params=objective_params,
        sf=sf,
        xp=xp,
    )
    _psf = rz_to_xyz(rz, (ny, nx), sf, off=xp.asarray(pos[:2]) / (dxy * 1e-6))
    if normalize:
        _psf /= xp.max(_psf)
    return _psf


def _centered_zv(nz: int, dz: float, pz: float = 0) -> npt.NDArray:
    lim = (nz - 1) * dz / 2
    return np.linspace(-lim + pz, lim + pz, nz)


def vectorial_psf_centered(
    nz: int,
    dz: float = 0.05,
    pz: float = 0,
    nx: int = 31,
    ny: int | None = None,
    pos: tuple[float, float, float] = (0, 0, 0),
    dxy: float = 0.05,
    wvl: float = 0.6,
    objective_params: ObjectiveKwargs | None = None,
    sf: int = 3,
    normalize: bool = True,
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
        objective_params=objective_params,
        sf=sf,
        normalize=normalize,
        xp=xp,
    )


# if __name__ == "__main__":
#     zv = np.linspace(-3, 3, 61)
#     from time import perf_counter

#     t0 = perf_counter()
#     psf = vectorial_psf(zv, nx=512)
#     t1 = perf_counter()
#     print(psf.shape)
#     print(t1 - t0)
#     assert np.allclose(np.load("out.npy"), psf, atol=0.1)
