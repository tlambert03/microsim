from collections.abc import Iterable, Sequence
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm import tqdm

from microsim._data_array import xrDataArray

try:
    import cupy as xp
    from cupyx.scipy.ndimage import map_coordinates
except ImportError:
    from scipy.ndimage import map_coordinates

    xp = np


class SIMIllum2D(BaseModel):
    angles: Sequence[float] = [0, np.pi / 3, np.pi * 2 / 3]
    nphases: int = 3
    linespacing: float = 0.2035
    defocus: float = 0  # TODO: implement
    spotsize: float = 0.035  # BFP spot size as fraction of NA
    nbeamlets: int = 31  # number of beamlets in each BFP spot
    ampcenter: float = 0
    order: str = "PZA"

    @property
    def nangles(self) -> int:
        return len(self.angles)

    @property
    def phaseshift(self) -> float:
        return 2 * self.linespacing / self.nphases

    @property
    def phases(self) -> NDArray:
        return np.arange(self.nphases) * self.phaseshift

    def axial_sim_plane(
        self,
        nz: int,
        nx: int,
        dz: float,
        dx: float,
        NA: float = 1.42,
        nimm: float = 1.515,
        wvl: float = 0.488,
    ) -> np.ndarray:
        """Return single axial SIM illumination plane."""
        return structillum_2d(  # type: ignore [no-any-return]
            nz=nz + 1,
            nx=nx,
            dz=dz,
            dx=dx,
            NA=NA,
            nimm=nimm,
            wvl=wvl,
            linespacing=self.linespacing,
            ampcenter=self.ampcenter,
            nbeamlets=self.nbeamlets,
            spotsize=self.spotsize,
        ).sum(0)[1:]

    def render(self, space: xrDataArray) -> xrDataArray:
        _dz = set(xp.round(xp.diff(space.coords.get("Z", [0, 0.1])), 8).tolist())
        _dx = set(xp.round(xp.diff(space.coords["X"]), 8).tolist())
        if len(_dz) != 1:
            raise ValueError("Non-uniform spacing detected in Z")
        if len(_dx) != 1:
            raise ValueError("Non-uniform spacing detected in X")
        dz = _dz.pop()
        dx = _dx.pop()
        nz = space.sizes.get("Z", 1)
        ny = space.sizes["Y"]
        nx = space.sizes["X"]

        data = self.create((nz, ny, nx), dz, dx)

        d = xr.DataArray(data, dims=list(self.order + "YX"), coords=space.coords)
        d.coords["A"] = self.angles
        d.coords["P"] = self.phases
        d.attrs["SIM"] = self.model_dump()
        return d

    def create(self, shape: tuple[int, int, int], dz: float, dx: float) -> np.ndarray:
        """Create illumination volume (generic variant of render)."""
        nz, ny, nx = shape

        nx_extended = int(3 * np.hypot(ny, nx))
        phases = self.phases / dx + nx_extended / 2

        # TODO: figure out the transformations without all the transpositions
        sim_plane = self.axial_sim_plane(nz=nz, nx=nx_extended, dz=dz, dx=dx).T

        out = np.empty((self.nangles, self.nphases, nz, ny, nx))

        coords = np.indices((ny, nz, nx)).reshape((3, -1))
        with tqdm(total=(self.nangles * self.nphases)) as pbar:
            for (ai, pi), (theta, phase) in _enumerated_product(self.angles, phases):
                pbar.set_description(
                    f"SIM: angle {ai + 1}/{self.nangles}, "
                    f"phase {pi + 1}/{self.nphases}"
                )
                img = self._render_plane(sim_plane, coords, theta, phase)
                img = img.reshape((ny, nz, nx)).transpose((1, 0, 2))
                pbar.update()
                out[ai, pi] = img.get() if hasattr(img, "get") else img

        _order = self.order.upper() + "YX"
        if _order != "APZYX":
            out = np.transpose(out, tuple("APZYX".index(i) for i in _order))
        return out

    def _render_plane(
        self, sim_plane: NDArray, coords: NDArray, theta: float, phase: float
    ) -> NDArray:
        if map_coordinates.__module__.startswith("cupy"):
            _i = []
            CHUNKSIZE = 128  # TODO: determine better strategy
            for chunk in np.array_split(coords, CHUNKSIZE, axis=1):
                new_coords = self._map_coords(xp.asarray(chunk), theta, phase)
                _i.append(map_coordinates(sim_plane, new_coords, order=1).get())
            img: np.ndarray = np.concatenate(_i)
        else:
            new_coords = self._map_coords(coords, theta, phase)
            img = map_coordinates(sim_plane, new_coords, order=1)
        return img

    def _map_coords(self, coords: NDArray, theta: float, phase: float) -> NDArray:
        """Map a set of img coordinates to new coords after phaseshift and rotation."""
        matrix = self._get_matrix(theta, phase)
        new_coordinates = (matrix[:-1, :-1] @ coords)[:2]
        return new_coordinates + xp.expand_dims(xp.asarray(matrix[:2, -1]), -1)  # type: ignore

    def _get_matrix(self, theta: float, phase: float) -> NDArray:
        """Get matrix to transform output coordinates to axial sim plane.

        Parameters
        ----------
        theta : float
            current rotation angle
        phase : float
            current phase shift

        Returns
        -------
        NDArray
            transformation matrix
        """
        scale = xp.eye(4)
        scale[2, 2] = 0  # flatten the z dimension to the 2D plane

        translate = xp.eye(4)
        translate[0, 3] = phase

        rotate = xp.array(
            [
                [np.cos(theta), 0, -np.sin(theta), 0],
                [0, 1, 0, 0],
                [np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )

        return scale @ translate @ rotate  # type: ignore


class SIMIllum3D(SIMIllum2D):
    nphases: int = 5
    ampcenter: float = 1.0


TAU = 1j * 2 * np.pi


def efield(kvec: tuple[float, float], zarr: NDArray, xarr: NDArray) -> NDArray:
    return xp.exp(TAU * (kvec[0] * xarr + kvec[1] * zarr))  # type: ignore


@lru_cache(maxsize=128)
def structillum_2d(
    nz: int = 128,
    nx: int = 256,
    dz: float = 0.01,
    dx: float = 0.01,
    NA: float = 1.42,
    nimm: float = 1.515,
    wvl: float = 0.488,
    linespacing: float = 0.2035,
    ampcenter: float = 1.0,
    ampratio: float = 1.0,  # TODO: remove?  ampcenter seems sufficient
    nbeamlets: int = 31,
    spotsize: float = 0.035,
) -> npt.NDArray:
    """Simulate a single "XZ" plane of structured illumination intensity.

    from Lin Shao's psfsimu.py file which is in turn based on Hanser et al (2004)
    and C code pftopsfV3test_polV2.c

    '2d' means we're only creating one axial sheet of illumination
    (not that we're doing 2D SIM)

    Parameters
    ----------
    nz : float, optional
        number of Z pixels
    nx : float, optional
        number of X pixels
    dz : float, optional
        pixel sizes in microns, by default 0.01
    dx : float, optional
        pixel sizes in microns, by default 0.01
    NA : float, optional
        NA of the lens, by default 1.42
    nimm : float, optional
        refractive index of immersion medium, by default 1.515
    wvl : float, optional
        wavelength in microns, by default 0.488
    linespacing : float, optional
        spacing in microns of illumination pattern, by default 0.2035
    ampcenter : float, optional
        the amplitude of the center illum beam if 0, then it's for 2D SIM.
        by default 1.0
    ampratio : float, optional
        the amplitude of the side beams relative to center beam, which is 1.0.
        by default 1.0
    nbeamlets : int, optional
        the number of triplets (or sextets) we'd divide the illumination beams
        into because the beams assume different incident angles (multi-mode fiber),
        by default 31
    spotsize : float, optional
        the proportion of illum spot size vs entire NA.  This represents the size of
        a partially coherent (e.g. multimode fiber) spot at the back focal plane.
        Increasing spotratio will reduce the axial envelope of the pattern.

    Returns
    -------
    np.ndarray
        (3 x nz x nx) Array of zero order, axial orders, and lateral orders. Sum them
        along axis zero to get final 3D illumination intensity.
    """
    if NA > nimm:
        raise ValueError("NA must be less than immersion refractive index `nimm`.")

    # steepest angle of illumination based on NA and nimm, in radians
    max_half_angle = xp.arcsin(NA / nimm)
    # angular span of each beam (where higher numbers mean a larger spot in the BFP)
    spot_angular_span = spotsize * 2 * max_half_angle
    # NA of each beam
    beam_NA = 0 if spot_angular_span == 0 else xp.sin(spot_angular_span)

    # NA_arr is the individual NAs for each wave-vector in each of the 2/3 beams
    # (each beam is composed of 'nbeamlets' beamlets spanning a total of NA_span)
    NA_arr: NDArray = xp.arange(nbeamlets, dtype=np.float32) - (nbeamlets - 1) / 2
    NA_arr *= beam_NA / nbeamlets

    # max k vector
    kmag = nimm / wvl

    # The contribution to the illum is dependent on theta, since the middle of
    # each illumination spot has more rays than the edge
    # ``kmag * beam_NA / 2`` is the radius of each circular illumination spot
    # `weight_arr` is essentially the "chord" length as a function of NA_arr
    _r = kmag * beam_NA / 2
    _d = kmag * NA_arr
    weight_arr = xp.sqrt(xp.maximum(_r**2 - _d**2, 0)) / _r

    side_NA = (1 / linespacing / 2 + kmag * NA_arr) / kmag
    if xp.any(side_NA > 1):
        # TODO could consider setting weight of clipped beams to 0?
        raise ValueError(
            "Unsatisfiable parameters (clipped beams): "
            f"{NA=}, {wvl=}, {linespacing=}, {spotsize=}"
        )

    linefrequency = 1 / linespacing
    # TODO: add sanity check here
    plus_sideNA = (linefrequency / 2 + kmag * NA_arr) / kmag
    minus_sideNA = -plus_sideNA[::-1]

    zarr, xarr = xp.indices((nz, nx)).astype(np.float32)
    zarr = (zarr - nz / 2) * dz
    xarr = (xarr - nx / 2) * dx

    kvecs = kmag * xp.stack([NA_arr, xp.sqrt(1 - NA_arr**2)]).T
    plus_kvecs = kmag * xp.stack([plus_sideNA, xp.sqrt(1 - plus_sideNA**2)]).T
    minus_kvecs = kmag * xp.stack([minus_sideNA, xp.sqrt(1 - minus_sideNA**2)]).T

    # output array
    intensity: NDArray = xp.zeros((3, nz, nx))
    for i, wght in enumerate(weight_arr):
        a0 = efield(kvecs[i], zarr, xarr) * ampcenter
        a1 = efield(plus_kvecs[i], zarr, xarr) * ampratio
        a2 = efield(minus_kvecs[i], zarr, xarr) * ampratio

        a1conj = a1.conj()
        a2conj = a2.conj()

        intensity[0] += xp.real(a0 * a0.conj() + a1 * a1conj + a2 * a2conj) * wght
        intensity[1] += 2 * xp.real(a0 * a1conj + a0 * a2conj) * wght
        intensity[2] += 2 * xp.real(a1 * a2conj) * wght

    return intensity


def _enumerated_product(
    *args: Any,
) -> Iterable[tuple[tuple[int, int], tuple[Any, ...]]]:
    yield from zip(
        product(*(range(len(x)) for x in args)), product(*args), strict=False
    )
