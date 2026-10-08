import warnings
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from annotated_types import Ge, Gt

from microsim._data_array import ArrayProtocol, DataArray, xrDataArray
from microsim._logger import logger, logging_indented
from microsim.psf import cached_spinning_disk_psf, make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.dimensions import Axis
from microsim.schema.lens import ObjectiveLens
from microsim.schema.settings import Settings
from microsim.schema.space import SpaceProtocol


class _PSFModality(SimBaseModel):
    # axes that are averaged (rather than summed) when downscaling to the output
    # space: a camera pixel integrates light over its area (sum), whereas each
    # point-scanning pixel, and each z-plane, is a measurement at one position.
    rescale_mean_axes: ClassVar[tuple[str, ...]] = ()
    # whether excitation saturation is applied locally in the PSF (in which case
    # the emission rates are left unsaturated)
    local_saturation: ClassVar[bool] = False

    def psf(
        self,
        *,
        nz: int,
        nx: int,
        dx: float,
        dz: float,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI,
        ex_wvl_nm: float | None = None,
        em_wvl_nm: float | None = None,
        saturation: float = 0,
    ) -> ArrayProtocol:
        # default implementation is a widefield PSF (uniform illumination, so
        # `saturation` is already applied to the emission rates and ignored here)
        return make_psf(
            nz=nz,
            nx=nx,
            dx=dx,
            dz=dz,
            objective=objective_lens,
            ex_wvl_nm=ex_wvl_nm,
            em_wvl_nm=em_wvl_nm,
            xp=xp,
        )

    def render(
        self,
        truth: xrDataArray,  # (F, Z, Y, X)
        em_rates: xrDataArray,  # (C, F, W)
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
    ) -> xrDataArray:
        """Render a 3D image of the truth for F fluorophores, in C channels."""
        # for every channel in the emission rates...
        channels = []
        for ch in em_rates.coords[Axis.C].values:
            logger.info(f"Rendering {type(self).__name__} channel {ch} ---------------")

            with logging_indented():
                # for every fluorophore in the sample...
                fluors = []
                for f_idx, fluor in enumerate(truth.coords[Axis.F].values):
                    logger.info(f"Fluor: {fluor}")
                    with logging_indented():
                        f_truth = truth.isel({Axis.F: f_idx})

                        # discretize the em spectrum for this specific ch/fluor pair
                        em_spectrum = em_rates.sel({Axis.C: ch, Axis.F: fluor})
                        # if we happen to have 2 spectra for the same fluorophore
                        # in the same channel, just take the first one (shouldn't happen
                        if Axis.F in em_spectrum.dims:  # pragma: no cover
                            em_spectrum = em_spectrum.isel({Axis.F: 0})

                        if not (em_spectrum > 1e-12).any():
                            # no emission at all for this fluorophore in this channel
                            fluors.append(xp.zeros_like(f_truth))
                            continue

                        summed_psf = self._summed_weighted_psf(
                            em_spectrum,
                            settings,
                            truth.attrs["space"],
                            objective_lens,
                            xp,
                        )
                        fluor_sum = xp.fftconvolve(f_truth, summed_psf, mode="same")
                        fluors.append(fluor_sum)

            # stack the fluorophores together to create the channel
            channels.append(xp.stack(fluors, axis=0))

        return DataArray(
            channels,
            dims=[Axis.C, Axis.F, Axis.Z, Axis.Y, Axis.X],
            coords={
                Axis.C: em_rates.coords[Axis.C],
                Axis.F: truth.coords[Axis.F],
                Axis.Z: truth.coords[Axis.Z],
                Axis.Y: truth.coords[Axis.Y],
                Axis.X: truth.coords[Axis.X],
            },
            attrs={
                "space": truth.attrs["space"],
                "objective": objective_lens,
                "units": "photons",
            },
        )

    def _summed_weighted_psf(
        self,
        em_spectrum: xrDataArray,
        settings: Settings,
        space: SpaceProtocol,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI,
    ) -> ArrayProtocol:
        """Create a weighted sum of PSFs based on the emission spectrum.

        This takes advantage of the distributive property of convolution
        (a * b) * c = a * (b * c)
        We create a PSF for each emission wavelength, multiply it by the
        emission rate at that wavelength, and sum them all together, prior
        to convolving with the truth.
        This creates a more realistic PSF for the fluorophore/channel, as it
        accounts for the full emission spectrum.
        """
        binned = bin_spectrum(
            em_spectrum,
            bins=settings.spectral_bins_per_emission_channel,
            threshold_percentage=settings.spectral_bin_threshold_percentage,
        )

        # we need to pick a single nx size for all psfs we will sum, based on the
        # maximum wavelength in the emission spectrum and `settings.max_psf_radius_aus`
        nz, _ny, _nx = space.shape
        dz, _dy, dx = space.scale
        max_wave = binned.coords[Axis.W].max().item()
        nx = _pick_nx(
            _nx,
            dx,
            settings.max_psf_radius_aus,
            max_wave,
            objective_lens.numerical_aperture,
        )

        saturation = self._saturation_parameter(em_spectrum)
        summed_psf: Any = 0
        for em_rate in binned:
            em_wvl_nm = em_rate.w.item()
            if em_rate.isnull().any() or em_rate == 0 or xp.isnan(em_wvl_nm):
                continue
            weight = em_rate.item()
            logger.info(f"Need PSF ({nz},{nx}) @ {em_wvl_nm:.1f} nm ({weight=:.2f})")
            psf = self.psf(
                nz=nz,
                nx=nx,
                dx=dx,
                dz=dz,
                objective_lens=objective_lens,
                em_wvl_nm=em_wvl_nm,
                saturation=saturation,
                xp=xp,
            )
            summed_psf += psf * weight
        return summed_psf  # type: ignore [no-any-return]

    def _saturation_parameter(self, em_spectrum: xrDataArray) -> float:
        """Saturation parameter to apply in the PSF (0 if not applied spatially)."""
        return 0.0


class Confocal(_PSFModality):
    """Point-scanning confocal.

    The PSF is the probability that a photon emitted by a fluorophore at a given
    position (relative to the scan spot) passes the pinhole, times the relative
    excitation intensity there.  Light source `power` is therefore interpreted as
    the peak irradiance at the focus, and `exposure_ms` as the per-pixel integration
    (dwell) time (it is not divided by the number of pixels).  Excitation saturation
    is applied locally (per position in the excitation PSF), rather than to the
    emission rates.
    """

    type: Literal["confocal"] = "confocal"
    rescale_mean_axes: ClassVar[tuple[str, ...]] = (Axis.Z, Axis.Y, Axis.X)
    local_saturation: ClassVar[bool] = True
    pinhole_au: Annotated[float, Ge(0)] = 1

    def _saturation_parameter(self, em_spectrum: xrDataArray) -> float:
        # emission rates are unsaturated for local_saturation modalities;
        # saturation is applied locally to the excitation PSF instead.
        oc = em_spectrum.coords[Axis.C].item()
        fluor = em_spectrum.coords[Axis.F].item()
        return _round_saturation(oc.saturation_parameter(fluor))

    def psf(
        self,
        *,
        nz: int,
        nx: int,
        dx: float,
        dz: float,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI,
        ex_wvl_nm: float | None = None,
        em_wvl_nm: float | None = None,
        saturation: float = 0,
    ) -> ArrayProtocol:
        return make_psf(
            nz=nz,
            nx=nx,
            dx=dx,
            dz=dz,
            objective=objective_lens,
            em_wvl_nm=em_wvl_nm,
            ex_wvl_nm=ex_wvl_nm,
            pinhole_au=self.pinhole_au,
            saturation=saturation,
            xp=xp,
        )


class SpinningDiskConfocal(_PSFModality):
    """Spinning-disk (Nipkow disk) confocal, imaged onto a camera.

    Pinhole positions follow the equal-pitch, multi-thread Archimedean spiral of a
    Yokogawa CSU disk (see `microsim.illum._spinning_disc`), and the time-averaged
    PSF includes crosstalk through neighboring pinholes (out-of-focus background in
    thick samples).  Neighbors are only included within the PSF window (see
    `Settings.max_psf_radius_aus`).

    The signal is collected by a camera: output pixels are summed over sub-pixels in
    xy (averaged in z), and `exposure_ms` is the camera exposure time.  Light source
    `power` is the *time-averaged* irradiance at the sample; each spot's peak
    irradiance is `power * pitch**2 / spot_area` (only relevant for saturation).

    Defaults are for a CSU-X1 (50 um pinholes, 5x spacing, ~20,000 pinholes on
    12 interleaved spirals).  Disk-plane sizes are projected onto the sample by
    `magnification` (objective x any relay optics).  The crosstalk pattern is that
    of the field center (it drifts slowly across the field).

    Approximations: saturation is per-spot (overlapping out-of-focus spots are not
    summed); finite microlens focal spot is ignored.
    """

    type: Literal["spinning_disk"] = "spinning_disk"
    local_saturation: ClassVar[bool] = True
    rescale_mean_axes: ClassVar[tuple[str, ...]] = (Axis.Z,)

    # total magnification from sample to the pinhole disk
    magnification: Annotated[float, Gt(0)] = 100
    pinhole_diameter_um: Annotated[float, Gt(0)] = 50  # on the disk
    pinhole_spacing_um: Annotated[float, Gt(0)] = 253  # on the disk
    disk_radii_mm: tuple[float, float] = (15, 25)  # inner/outer pinhole radii
    frames_per_rev: Annotated[float, Gt(0)] = 12  # interleaved spirals
    # effective NA of each excitation beamlet (scan heads typically underfill the
    # objective; not published by Yokogawa).  None = objective NA.
    excitation_na: Annotated[float, Gt(0)] | None = None

    def _saturation_parameter(self, em_spectrum: xrDataArray) -> float:
        # `s` at the time-averaged irradiance (converted to peak in the PSF)
        oc = em_spectrum.coords[Axis.C].item()
        s = oc.saturation_parameter(em_spectrum.coords[Axis.F].item())
        return _round_saturation(s, cutoff=1e-6)

    def psf(
        self,
        *,
        nz: int,
        nx: int,
        dx: float,
        dz: float,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI,
        ex_wvl_nm: float | None = None,
        em_wvl_nm: float | None = None,
        saturation: float = 0,
    ) -> ArrayProtocol:
        ex_wvl_nm = ex_wvl_nm or em_wvl_nm
        em_wvl_nm = em_wvl_nm or ex_wvl_nm
        if ex_wvl_nm is None or em_wvl_nm is None:
            raise ValueError("Either ex_wvl_nm or em_wvl_nm must be provided.")
        return cached_spinning_disk_psf(
            nz=nz,
            nx=nx,
            dx=dx,
            dz=dz,
            ex_wvl_um=ex_wvl_nm * 1e-3,
            em_wvl_um=em_wvl_nm * 1e-3,
            objective=objective_lens,
            pinhole_diameter_um=self.pinhole_diameter_um,
            pinhole_spacing_um=self.pinhole_spacing_um,
            disk_radii_mm=self.disk_radii_mm,
            frames_per_rev=self.frames_per_rev,
            magnification=self.magnification,
            saturation=saturation,
            excitation_na=self.excitation_na,
            xp=xp,
        )


def _round_saturation(s: float, cutoff: float = 1e-4) -> float:
    # ignore negligible saturation (reuses the unsaturated PSF), and round to
    # keep PSF cache keys stable
    return float(f"{s:.4g}") if s > cutoff else 0.0


class Widefield(_PSFModality):
    type: Literal["widefield"] = "widefield"


class Identity(_PSFModality):
    """Optical modality in which PSF is not applied.

    The idea is to use this modality when the ground truth flurophore distribution is
    generated from a light micorscope image, i.e., the PSF convolution is already
    applied on the image.  This is useful primarily when you are more interested in the
    spectral properties (fluorophores, filters, bleedthrough, etc.) than the spatial
    properties (PSF, modality, etc.) in the simulation.
    """

    def render(
        self,
        truth: xrDataArray,  # (F, Z, Y, X)
        em_rates: xrDataArray,  # (C, F, W)
        *args: Any,
        **kwargs: Any,
    ) -> xrDataArray:
        """Render a 3D image of the truth for F fluorophores, in C channels.

        In this case we don't apply the PSF convolution, as the truth is assumed to be
        already convolved with the PSF. Therefore, we simply compute the emission flux
        for each fluorophore and each channel.
        """
        em_image = em_rates.sum(Axis.W) * truth
        return DataArray(
            em_image,
            dims=[Axis.C, Axis.F, Axis.Z, Axis.Y, Axis.X],
            coords={
                Axis.C: em_rates.coords[Axis.C],
                Axis.F: truth.coords[Axis.F],
                Axis.Z: truth.coords[Axis.Z],
                Axis.Y: truth.coords[Axis.Y],
                Axis.X: truth.coords[Axis.X],
            },
            attrs={
                "space": truth.attrs["space"],
                "objective": "",
                "units": "photons",
            },
        )


def bin_spectrum(
    spectrum: xrDataArray,
    bins: int | np.ndarray = 3,
    *,
    threshold_intensity: float | None = None,
    threshold_percentage: float | None = None,
    max_bin_length: float | None = None,
    min_bin_length: float | None = None,
) -> xrDataArray:
    # Filter the spectrum to include only the region of interest
    # (where intensity is significant)
    if threshold_percentage is not None:
        if threshold_intensity is not None:
            warnings.warn(
                "Both threshold_intensity and threshold_percentage are provided. "
                "Only threshold_percentage will be used.",
                stacklevel=2,
            )
        mask = spectrum.values > (threshold_percentage * spectrum.values.max() / 100)
    elif threshold_intensity is not None:
        mask = spectrum.values > threshold_intensity
    else:
        mask = slice(None)
    masked = spectrum[mask]

    # pick bins, aiming for num_bins, but resulting in
    # a bin length of at least min_bin_length and at most max_bin_length
    if isinstance(bins, int):
        num_bins = bins
        # cast to float: passing 0-d xarray DataArrays to np.linspace triggers
        # __array_wrap__ on numpy >=2, which rewraps the result into a Variable
        # with mismatched dims.
        w_min, w_max = float(masked.w.min()), float(masked.w.max())
        w_range = w_max - w_min
        bin_length = w_range / num_bins
        if max_bin_length is not None:
            num_bins = max(num_bins, int(w_range / max_bin_length))
            bin_length = w_range / num_bins
        if min_bin_length and bin_length < min_bin_length:
            num_bins = int(w_range / min_bin_length)
            bin_length = w_range / num_bins
        bins = np.linspace(w_min, w_max, num_bins + 1)

    # Use groupby_bins to bin the data within the filtered region
    binned = masked.groupby_bins(Axis.W, bins=bins)

    # Create a new DataArray with the summed intensities and centroid wavelengths
    binned_spectrum = binned.sum(Axis.W)

    # Calculate the centroid wavelength for each bin
    centroids = binned.map(lambda x: (x.w * x).sum() / x.sum())
    binned_spectrum.coords.update({Axis.W: centroids})
    # Swap the dimensions to make the wavelength centroid the primary dimension
    binned_spectrum = binned_spectrum.swap_dims({"w_bins": Axis.W})
    return binned_spectrum


def _pick_nx(
    nx: int, dx: float, max_au_relative: float | None, ex_wvl_um: float, na: float
) -> int:
    # now restrict nx to no more than max_au_relative
    if max_au_relative is not None:
        airy_radius = 0.61 * ex_wvl_um / na
        n_pix_per_airy_radius = airy_radius / dx
        max_nx = int(n_pix_per_airy_radius * max_au_relative * 2)
        nx = min(nx, max_nx)
        # if even make odd
        if nx % 2 == 0:
            nx += 1
    return nx
