from collections.abc import Iterable, Sequence
from functools import cache
from itertools import product
from typing import TYPE_CHECKING, Annotated, Any, cast

import numpy as np
import numpy.typing as npt
from annotated_types import Ge
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr
from tqdm import tqdm

from microsim._data_array import ArrayProtocol, DataArray
from microsim.psf import vectorial_psf_centered
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings
from microsim.schema.space import Space

if TYPE_CHECKING:
    from microsim.schema.space import SpaceProtocol


class SIM2D(BaseModel):
    angles: Sequence[float] = [0, np.pi / 3, np.pi * 2 / 3]
    nphases: int = 3
    linespacing: float = 0.2035
    defocus: float = 0  # TODO: implement
    spotsize: Annotated[float, Ge(0)] = 0.035  # BFP spot size as fraction of NA
    nbeamlets: int = 31  # number of beamlets in each BFP spot
    ampcenter: float = 0
    order: str = "PZA"

    _xp: NumpyAPI = PrivateAttr()

    @classmethod
    def supports_optical_image(cls) -> bool:
        return False

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
        na: float = 1.42,
        nimm: float = 1.515,
        wvl: float = 0.488,
    ) -> np.ndarray:
        """Return single axial SIM illumination plane."""
        return structillum_2d(  # type: ignore[no-any-return]
            nz=nz + 1,
            nx=nx,
            dz=dz,
            dx=dx,
            NA=na,
            nimm=nimm,
            wvl=wvl,
            linespacing=self.linespacing,
            ampcenter=self.ampcenter,
            nbeamlets=self.nbeamlets,
            spotsize=self.spotsize,
            xp=self._xp,
        ).sum(0)[1:]

    def digital_image(
        self,
        truth: DataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        outspace: Space,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        # out_shape, upscale_xy, upscale_z, illum, truth, psf
        self._xp = xp = NumpyAPI.create(xp)

        space = cast("SpaceProtocol", truth.attrs["space"])
        truth_nz = truth.sizes.get("Z", 1)
        truth_ny = truth.sizes["Y"]
        truth_nx = truth.sizes["X"]
        truth_dz = space.scales["Z"]
        truth_dx = space.scales["X"]
        ex_wvl_um = channel.excitation.bandcenter * 1e-3
        em_wvl_um = channel.emission.bandcenter * 1e-3

        illum = self.illum_volume(
            (truth_nz * 2, truth_ny, truth_nx),
            truth_dz,
            truth_dx,
            na=objective_lens.numerical_aperture,
            wvl=ex_wvl_um,
            nimm=objective_lens.numerical_aperture,
        )
        # normalize?
        illum -= xp.min(illum)
        illum *= 1 / xp.max(illum)
        # adjust contrast  TODO
        # illum *= max(0, min(illum_contrast, 1))
        # illum += 1 - xp.max(illum)

        # coords = {}
        # for dim in "APZYX":
        #     if dim == "A":
        #         coords[dim] = self.angles
        #     elif dim == "P":
        #         coords[dim] = list(self.phases)
        #     else:
        #         coords[dim] = space.coords[dim]

        # these make sure that the psf is generated with an odd number of pixels
        # while np.pad at the bottom will retain the original shape
        trimz = int(not truth_nz % 2)
        trimx = int(not truth_nx % 2)
        psf = vectorial_psf_centered(
            truth_nz - trimz,
            nx=truth_nx - trimx,
            dz=truth_dz,
            dxy=truth_dx,
            pz=0,
            wvl=em_wvl_um,
            objective=objective_lens,
            xp=xp,
        )
        psf /= psf.sum()
        psf = np.pad(psf, ((trimz, 0), (trimx, 0), (trimx, 0)))

        yield from self._do_sim(outspace, illum, truth, psf)

    def _do_sim(
        self,
        outspace: Space,
        illum: ArrayProtocol,
        truth: DataArray,
        psf: ArrayProtocol,
    ) -> Iterable[DataArray]:
        # TODO clean this up...
        z_scale_factor = outspace.scales["Z"] / truth.attrs["space"].scales["Z"]
        if z_scale_factor % 1 != 0:
            raise ValueError("Z scale factor must be an integer")
        z_scale_factor = int(z_scale_factor)
        truth_nz = truth.sizes.get("Z", 1)
        illum = illum.astype(np.float32)
        nangles, nphases = illum.shape[:2]
        nplanes = outspace.sizes["Z"]
        desc = f"plane: {{}}/{nplanes}, angle: {{}}/{nangles}, phase: {{}}/{nphases}"
        xp = self._xp
        otf = xp.fft.rfftn(psf).astype(np.complex64)
        # FIXME: loosen dtype requirement
        with tqdm(total=nangles * nphases * nplanes) as pbar:
            for plane in range(nplanes):
                start = plane * z_scale_factor
                need_plane = truth_nz - (z_scale_factor // 2 + start) - 1
                for a, p in product(range(nangles), range(nphases)):
                    pbar.set_description(desc.format(plane + 1, a + 1, p + 1))
                    # get the illumination pattern for this angle and phase and plane
                    illum_chunk = illum[a, p, start : start + truth_nz]
                    # generate emission density
                    emission = truth * illum_chunk
                    # apply optical transfer function
                    emission = xp.fft.irfftn(xp.fft.rfftn(emission) * otf)
                    yield xp.real(xp.fft.fftshift(emission)[need_plane])
                    pbar.update()

    # def render(
    #     self,
    #     truth: DataArray,
    #     channel: "OpticalConfig",
    #     objective_lens: "ObjectiveLens",
    #     settings: "Settings",
    #     xp: NumpyAPI | None = None,
    # ) -> DataArray:
    #     self._xp = xp = NumpyAPI.create(xp)

    #     space = cast("SpaceProtocol", truth.attrs["space"])
    #     nz = truth.sizes.get("Z", 1)
    #     ny = truth.sizes["Y"]
    #     nx = truth.sizes["X"]
    #     dz = space.scales["Z"]
    #     dx = space.scales["X"]

    #     illum_vol = self.illum_volume((nz, ny, nx), dz, dx)
    #     emission = truth * illum_vol

    #     attrs = {**truth.attrs, "SIM": self.model_dump()}
    #     return DataArray(emission.data, coords=emission.coords, attrs=attrs)

    def illum_volume(
        self,
        shape: tuple[int, int, int],
        dz: float,
        dx: float,
        na: float = 1.42,
        wvl: float = 0.488,
        nimm: float = 1.515,
    ) -> ArrayProtocol:
        """Create illumination volume."""
        nz, ny, nx = shape

        # nx_extended is the longest possible diagonal in the XY plane
        # we calculate this to ensure that the illumination pattern is large enough
        # for all possible rotations
        nx_extended = int(3 * np.hypot(ny, nx))
        phases = self.phases / dx + nx_extended / 2

        # TODO: figure out the transformations without all the transpositions
        sim_plane = self.axial_sim_plane(
            nz=nz, nx=nx_extended, dz=dz, dx=dx, na=na, wvl=wvl, nimm=nimm
        ).T

        # coordinates for each point in the final rendered volume
        plane_shape = (ny, nz, nx)
        volume_coords = np.indices(plane_shape).reshape((3, -1))

        # generate each angle and phase combination
        data = np.empty((self.nangles, self.nphases, nz, ny, nx))
        with tqdm(total=(self.nangles * self.nphases)) as pbar:
            for (ai, pi), (theta, phase) in _enumerated_product(self.angles, phases):
                d = f"SIM: angle {ai + 1}/{self.nangles}, phase {pi + 1}/{self.nphases}"
                pbar.set_description(d)
                img = self._render_sim_volume(sim_plane, volume_coords, theta, phase)
                img = img.reshape(plane_shape).transpose((1, 0, 2))  # YZX -> ZYX
                pbar.update()
                data[ai, pi] = img.get() if hasattr(img, "get") else img

        return data

    def _render_sim_volume(
        self,
        sim_plane: NDArray,
        coords: NDArray,
        theta: float,
        phase: float,
    ) -> NDArray:
        """Take a single RZ SIM plane and map it to the 3D volume."""
        xp = self._xp
        if xp.map_coordinates.__module__.startswith("cupy"):
            _i = []
            CHUNKSIZE = 128  # TODO: determine better strategy
            for chunk in np.array_split(coords, CHUNKSIZE, axis=1):
                new_coords = self._map_coords(xp.asarray(chunk), theta, phase)
                _i.append(xp.map_coordinates(sim_plane, new_coords, order=1).get())
            img: np.ndarray = np.concatenate(_i)
        else:
            new_coords = self._map_coords(coords, theta, phase)
            img = xp.map_coordinates(sim_plane, new_coords, order=1)
        return img

    def _map_coords(self, coords: NDArray, theta: float, phase: float) -> NDArray:
        """Map a set of img coordinates to new coords after phaseshift and rotation."""
        matrix = self._get_matrix(theta, phase)
        new_coordinates = (matrix[:-1, :-1] @ coords)[:2]
        xp = self._xp
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
            tranformation matrix
        """
        xp = self._xp
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


class SIM3D(SIM2D):
    nphases: int = 5
    ampcenter: float = 1.0


TAU = 1j * 2 * np.pi


def efield(
    kvec: tuple[float, float], zarr: NDArray, xarr: NDArray, xp: NumpyAPI | None = None
) -> NDArray:
    """
    Calculate the electric field at each point in space.

    Parameters
    ----------
    kvec : tuple[float, float]
        A tuple representing the wave vector components.
    zarr : NDArray
        An array representing the z-coordinate values.
    xarr : NDArray
        An array representing the x-coordinate values.
    xp : NumpyAPI, optional
        The numpy backend to use, by default will use numpy.

    Returns
    -------
    NDArray
        An array representing the electric field at each point in space.
    """
    return (xp or np).exp(TAU * (kvec[0] * xarr + kvec[1] * zarr))  # type: ignore


@cache
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
    xp: NumpyAPI | None = None,
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
    xp : NumpyAPI, optional
        The numpy backend to use, by default will use numpy.

    Returns
    -------
    np.ndarray
        (3 x nz x nx) Array of zero order, axial orders, and lateral orders. Sum them
        along axis zero to get final 3D illumination intensity.
    """
    if NA > nimm:
        raise ValueError("NA must be less than immersion refractive index `nimm`.")

    xp = NumpyAPI.create(xp)
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
