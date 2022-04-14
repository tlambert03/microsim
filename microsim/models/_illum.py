from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from pydantic import BaseModel

from ._renderable import Renderable

try:
    import cupy as xp
except ImportError:
    xp = np

if TYPE_CHECKING:
    import napari.types
    import xarray as xr
    from numpy.typing import NDArray


class Illumination(Renderable):
    ...


class SIMIllum2D(Illumination, BaseModel):
    angles: Sequence[float] = [0, np.pi / 3, np.pi * 2 / 3]
    nphases: int = 3
    linespacing: float = 0.2035
    defocus: float = 0

    @property
    def nangles(self) -> int:
        return len(self.angles)

    @property
    def phaseshift(self) -> float:
        return 2 * self.linespacing / self.nphases


class SIMIllum3D(SIMIllum2D):
    nphases: int = 5

    def render(self, space: xr.DataArray):
        assert space.ndim == 3
        nz, ny, nx = space.shape  # Todo, use xarray


def efield(kvec, zarr, xarr, dx, dz):
    return np.exp(1j * 2 * np.pi * (kvec[0] * xarr * dx + kvec[1] * zarr * dz))


def structillum_2d(
    nz: int = 256,
    nx: int = 512,
    dx: float = 0.01,
    dz: float = 0.01,
    NA: float = 1.42,
    nimm: float = 1.515,
    wvl: float = 0.488,
    linespacing: float = 0.2035,
    extraz: int = 0,
    side_intensity: float = 0.5,
    ampcenter: float = 1.0,
    ampratio: float = 1.0,
    nangles: int = 100,
    spotratio: float = 0.035,
) -> napari.types.ImageData:
    """Simulate a single "XZ" plane of structured illumination intensity.

    from Lin Shao's psfsimu.py file which is in turn based on Hanser et al (2004) and
    C code pftopsfV3test_polV2.c

    '2d' means we're only creating one sheet of illumination since every sheet will
    be the same side_intensity (0~1) -- the amplitude of illum from one objective;
    for the other it's 1 minus this value

    Parameters
    ----------
    shape : _type_
        output array shape
    dx : float, optional
        pixel sizes in microns, by default 0.01
    dz : float, optional
        pixel sizes in microns, by default 0.01
    NA : float, optional
        NA of the lens, by default 1.42
    nimm : float, optional
        refractive index of immersion medium, by default 1.515
    wvl : float, optional
        wavelength in microns, by default 0.488
    linespacing : float, optional
        spacing in microns of illumination pattern, by default 0.2035
    extraz : int, optional
        _description_, by default 0
    side_intensity : float, optional
        the amplitude of illum from one objective; for the other it's 1 minus this
        value. Should be between 0-1. by default 0.5
    ampcenter : float, optional
        the amplitude of the center illum beam if 0, then it's for 2D SIM.
        by default 1.0
    ampratio : float, optional
        the amplitude of the side beams relative to center beam, which is 1.0.
        by default 1.0
    nangles : int, optional
        the number of triplets (or sextets) we'd divide the illumination beams
        into because the beams assume different incident angles (multi-mode fiber),
        by default 100
    spotratio : float, optional
        the proportion of illum spot size vs entire NA with the condition that
        f-fiber-lens = 60mm, fiber-diameter = 0.1mm, lambda-excitation = 528nm,
        grating-pitch = 33.6um, by default 0.035

    Returns
    -------
    _type_
        _description_
    """

    # theta_arr = np.arange(-nangles/2, nangles/2+1, dtype=np.float32)*anglespan/nangles

    # NA is half angle, hence 2 *
    anglespan = spotratio * 2 * np.arcsin(NA / nimm)

    NA_span = np.sin(anglespan)
    NA_arr = (
        np.arange(-nangles / 2, nangles / 2 + 1, dtype=np.float32) * NA_span / nangles
    )

    kmag = nimm / wvl

    # The contribution to the illum is dependent on theta, since the middle of
    # the circle has more rays than the edge
    # kmag*np.sin(anglespan/2)) is the radius of each circular illumination
    # spot weight_arr is essentially the "chord" length as a function of theta_arr
    # weight_arr = np.sqrt(
    #   (kmag*np.sin(anglespan/2)) ** 2 - (kmag*np.sin(theta_arr))**2 )
    #   / (kmag*np.sin(anglespan/2))
    weight_arr = np.sqrt((kmag * NA_span / 2) ** 2 - (kmag * NA_arr) ** 2) / (
        kmag * NA_span / 2
    )

    # plus_sidetheta_arr =
    #   np.arcsin( (kmag * np.sin(theta_arr) + 1/linespacing/2)/kmag)
    # minus_sidetheta_arr = -plus_sidetheta_arr[::-1]

    linefrequency = 1 / linespacing
    # TODO: add sanity check here
    plus_sideNA = (linefrequency / 2 + kmag * NA_arr) / kmag
    minus_sideNA = -plus_sideNA[::-1]

    # output array
    intensity: NDArray = np.zeros((3, nz + extraz, nx), np.float32)

    amp: NDArray = np.zeros((3, nz + extraz, nx), np.complex64)
    zarr, xarr = np.indices((nz + extraz, nx)).astype(np.float32)
    zarr -= (nz + extraz) / 2
    xarr -= nx / 2

    amp_plus = np.sqrt(1.0 - side_intensity)
    # amp_minus = np.sqrt(side_intensity)

    kvecs = kmag * np.stack([NA_arr, np.sqrt(1 - NA_arr**2)]).T
    plus_kvecs = kmag * np.stack([plus_sideNA, np.sqrt(1 - plus_sideNA**2)]).T
    minus_kvecs = kmag * np.stack([minus_sideNA, np.sqrt(1 - minus_sideNA**2)]).T

    for i, wght in enumerate(weight_arr):
        amp[0] = amp_plus * efield(kvecs[i], zarr, xarr, dx, dz) * ampcenter
        amp[1] = amp_plus * efield(plus_kvecs[i], zarr, xarr, dx, dz) * ampratio
        amp[2] = amp_plus * efield(minus_kvecs[i], zarr, xarr, dx, dz) * ampratio

        intensity[0] += (
            (amp[0] * amp[0].conj() + amp[1] * amp[1].conj() + amp[2] * amp[2].conj())
            * wght
        ).real
        intensity[1] += (
            2 * np.real(amp[0] * amp[1].conj() + amp[0] * amp[2].conj()) * wght
        )
        intensity[2] += 2 * np.real(amp[1] * amp[2].conj()) * wght

    if extraz <= 0:
        return intensity

    # blend = F.zeroArrF(extraz, nx)
    aslope = np.arange(extraz, dtype=np.float32) / extraz
    blend = np.transpose(
        np.transpose(intensity[:extraz, :]) * aslope
        + np.transpose(intensity[-extraz:, :]) * (1 - aslope)
    )
    intensity[:extraz, :] = blend
    intensity[-extraz:, :] = blend
    return intensity[extraz // 2 : -extraz // 2, :]  # noqa
