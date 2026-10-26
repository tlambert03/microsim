from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
from annotated_types import Ge
from pydantic import Field

from microsim._data_array import DataArray, xrDataArray
from microsim._logger import logger, logging_indented
from microsim._psf import PSFGrid, airy_radius_um, apply_saturation
from microsim.fft.convolve import fftconvolve_same_pairs
from microsim.schema._base_model import SimBaseModel
from microsim.schema.dimensions import Axis

from ._optics import (
    CameraDetection,
    Detection,
    FocusedIllumination,
    Illumination,
    PinholeDetection,
    SpectralLines,
    WidefieldIllumination,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from microsim._data_array import ArrayProtocol
    from microsim.schema.backend import NumpyAPI
    from microsim.schema.lens import ObjectiveLens
    from microsim.schema.optical_config._fluorescence import ChannelFluorescence
    from microsim.schema.settings import Settings
    from microsim.schema.space import SpaceProtocol


class _PSFModality(SimBaseModel):
    # how the image is detected (see `digital_image`, for how pixels are formed)
    detection_type: ClassVar[type[Detection]] = CameraDetection

    def _illumination(self) -> Illumination:
        return WidefieldIllumination()

    def _detection(
        self,
        fluorescence: Sequence[Sequence[ChannelFluorescence]],
        c_idx: int,
        objective: ObjectiveLens,
    ) -> Detection:
        """Detection for channel `c_idx` (`fluorescence` is `[channel][fluor]`)."""
        return CameraDetection()

    def render(
        self,
        truth: xrDataArray,  # (F, Z, Y, X)
        fluorescence: Sequence[Sequence[ChannelFluorescence]],  # [channel][fluor]
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
    ) -> xrDataArray:
        """Render a 3D image of the truth for F fluorophores, in C channels.

        `fluorescence[c]` describes each fluorophore in channel `c`.  The result is
        in photons/s (detected), with dimensions (C, F, Z, Y, X).
        """
        space = truth.attrs["space"]
        # system PSF (and brightness) for every channel/fluorophore pair
        system_psfs: dict[tuple[int, int], tuple[Hashable, ArrayProtocol]] = {}
        brightness: dict[tuple[int, int], float] = {}
        for c_idx, channel in enumerate(fluorescence):
            logger.info(f"Rendering {type(self).__name__} channel {channel[0].channel}")
            with logging_indented():
                detection: Detection | None = None
                by_fluor = {fl.fluorophore: fl for fl in channel}
                for f_idx, fluor in enumerate(truth.coords[Axis.F].values):
                    fl = by_fluor[fluor]
                    if fl.detected_rate(saturate=False) <= 1e-12:
                        continue  # this fluorophore is not seen in this channel
                    if detection is None:  # (only needed if something is seen)
                        detection = self._detection(fluorescence, c_idx, objective_lens)
                    logger.info(f"Fluor: {fluor}")
                    with logging_indented():
                        key, psf, bright = self._system_psf(
                            fl, detection, space, objective_lens, settings, xp
                        )
                    system_psfs[(c_idx, f_idx)] = (key, psf)
                    brightness[(c_idx, f_idx)] = bright

        # convolve each fluorophore's truth with its PSF in each channel -> [c][f]
        images = fftconvolve_same_pairs(
            [truth.data[f_idx] for f_idx in range(truth.sizes[Axis.F])],
            system_psfs,
            n_out=len(fluorescence),
            xp=xp,
            cache_mb=settings.cache.in_mem_size.otf_mb,
        )
        for (c_idx, f_idx), bright in brightness.items():
            images[c_idx][f_idx] = images[c_idx][f_idx] * bright
        return DataArray(
            [xp.stack(fluors, axis=0) for fluors in images],
            dims=[Axis.C, Axis.F, Axis.Z, Axis.Y, Axis.X],
            coords={
                Axis.C: [channel[0].channel for channel in fluorescence],
                Axis.F: truth.coords[Axis.F],
                Axis.Z: truth.coords[Axis.Z],
                Axis.Y: truth.coords[Axis.Y],
                Axis.X: truth.coords[Axis.X],
            },
            attrs={
                "space": space,
                "objective": objective_lens,
                "units": "photons",
            },
        )

    def _system_psf(
        self,
        fl: ChannelFluorescence,
        detection: Detection,
        space: SpaceProtocol,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
    ) -> tuple[Hashable, ArrayProtocol, float]:
        """Return the system PSF of one fluorophore in one channel, and its brightness.

        The system PSF is the excitation spot (relative excitation, 1 at the focus)
        times the detection PSF (fraction of collected light detected).  The image of
        the fluorophore is `brightness * (truth (*) system_psf)`, where `brightness`
        is in detected photons/s per fluorophore.  Also returns a key that uniquely
        identifies the system PSF, for caching.

        Emission is spread over several wavelengths (see
        `settings.spectral_bins_per_emission_channel`), so the detection PSF is a
        weighted sum of PSFs at each.  For focused illumination, the excitation spot
        is likewise a weighted sum over the excitation wavelengths.
        """
        emission = emission_lines(
            fl.detected_fraction,
            bins=settings.spectral_bins_per_emission_channel,
            threshold_percentage=settings.spectral_bin_threshold_percentage,
        )
        excitation = excitation_lines(
            fl.absorption_rate,
            threshold_percentage=settings.spectral_bin_threshold_percentage,
            bin_nm=settings.excitation_bin_width_nm,
        )
        grid = self._psf_grid(space, emission, detection, objective_lens, settings)
        # round, to reuse the PSF for (practically) identical saturation levels
        saturation = float(f"{fl.saturation:.4g}") if fl.saturation > 1e-4 else 0.0

        logger.info(f"Need PSF {grid} @ emission {emission} nm")
        detected: Any = None
        for wvl_nm, weight in emission:
            psf = detection.psf(wvl_nm, grid, objective_lens, xp) * weight
            if detected is None:
                detected = psf  # a new array (cached PSFs are read-only)
            else:
                detected += psf

        key: tuple = (detection, grid, emission, objective_lens.cache_key(), type(xp))
        spot = self._illumination().spot(excitation, grid, objective_lens, xp)
        if isinstance(spot, float):
            # uniform illumination: excitation (and saturation) is just a factor
            brightness = fl.detected_rate(saturate=False) * apply_saturation(
                spot, fl.saturation
            )
            return key, detected, brightness
        key = (*key, excitation, saturation)
        # apply_saturation(spot, saturation) * detected, without temporary copies
        detected *= spot
        if saturation:
            denominator: Any = spot * saturation
            denominator += 1
            detected /= denominator
        return key, detected, fl.detected_rate(saturate=False)

    def _psf_grid(
        self,
        space: SpaceProtocol,
        emission: SpectralLines,
        detection: Detection,
        objective_lens: ObjectiveLens,
        settings: Settings,
    ) -> PSFGrid:
        # one PSF size for all wavelengths, based on the longest emission wavelength
        # and `settings.max_psf_radius_aus`
        nz, _ny, _nx = space.shape
        dz, _dy, dx = space.scale
        max_wave_um = max(wvl for wvl, _ in emission) / 1000
        radius_aus = settings.max_psf_radius_aus
        if radius_aus == "auto":
            radius_aus = detection.psf_radius_aus(nz * dz, max_wave_um, objective_lens)
            logger.info(f"Auto PSF radius: {radius_aus:.1f} AU")
        na = objective_lens.numerical_aperture
        nx = _pick_nx(_nx, dx, radius_aus, max_wave_um, na)
        return PSFGrid(nz=nz, nx=nx, dz=dz, dx=dx)


class Confocal(_PSFModality):
    """Point-scanning confocal.

    The PSF is the probability that a photon emitted by a fluorophore at a given
    position (relative to the scan spot) passes the pinhole, times the relative
    excitation intensity there.  The channel's `power` (`OpticalConfig.power`, in
    W/cm^2) is therefore interpreted as the peak irradiance at the focus, and
    `exposure_ms` as the per-pixel integration (dwell) time (it is not divided by
    the number of pixels).  The excitation spot is
    computed at the excitation wavelengths, and excitation saturation is applied
    locally (it is strongest at the center of the spot).

    Parameters
    ----------
    pinhole_au : float
        Pinhole diameter in Airy units (1 AU = 1.22 * wavelength / NA, at the
        wavelength set by `airy_unit_wavelength_nm`).
    """

    type: Literal["confocal"] = "confocal"
    # a (nearly) closed pinhole passes (almost) no light.  Real scanners rarely go
    # below ~0.2 AU.
    pinhole_au: Annotated[float, Ge(0.05)] = 1
    airy_unit_wavelength_nm: float | None = Field(
        default=None,
        gt=0,
        description="Wavelength (nm) at which `pinhole_au` is defined.  By default, "
        "1 AU is matched to the average emission filter band center of the channels "
        "sharing the pinhole (or, if not `shared_pinhole`, to each channel's band "
        "center).",
    )
    shared_pinhole: bool = Field(
        default=True,
        description="If True (the default), channels acquired together (in the same "
        "track, see `Simulation.tracks`) are detected through the same pinhole, as "
        "on many point scanners; the pinhole is set anew for each track.  If False, "
        "each channel has its own pinhole, matched to its emission band.",
    )

    detection_type = PinholeDetection

    def _illumination(self) -> Illumination:
        return FocusedIllumination()

    def _detection(
        self,
        fluorescence: Sequence[Sequence[ChannelFluorescence]],
        c_idx: int,
        objective: ObjectiveLens,
    ) -> Detection:
        # the channels sharing this channel's pinhole: those in the same track
        track = fluorescence[c_idx][0].track
        if self.shared_pinhole and track is not None:
            group = [row for row in fluorescence if row[0].track == track]
        else:
            group = [fluorescence[c_idx]]
        wvl_nm = self.airy_unit_wavelength_nm
        if wvl_nm is None:
            centers = [row[0].channel.emission_band_center() for row in group]
            if None not in centers:
                wvl_nm = float(np.mean(centers))  # type: ignore [arg-type]
        if wvl_nm is None:
            # no band-limited emission filter (e.g. a longpass): use the center of
            # the light detected in these channel(s) instead
            wvl_nm = _detected_light_center([fl for row in group for fl in row])
            names = ", ".join(repr(row[0].channel.name) for row in group)
            # (warnings are shown once per distinct message)
            warnings.warn(
                f"Channel(s) {names} have no band-limited emission filter to match "
                f"the pinhole size to; using 1 AU at {wvl_nm:.0f} nm (the center of "
                "the detected emission).  Set `Confocal.airy_unit_wavelength_nm` to "
                "choose.",
                stacklevel=2,
            )
        radius_um = self.pinhole_au * airy_radius_um(
            wvl_nm / 1000, objective.numerical_aperture
        )
        # round, to keep PSF cache keys stable
        return PinholeDetection(radius_um=float(f"{radius_um:.6g}"))


class Widefield(_PSFModality):
    """Widefield: uniform illumination, imaged onto a camera.

    The optical image is the emission of each fluorophore convolved with the
    emission PSF.
    """

    type: Literal["widefield"] = "widefield"


class Identity(_PSFModality):
    """Optical modality in which PSF is not applied.

    The idea is to use this modality when the ground truth fluorophore distribution
    is generated from a light microscope image, i.e., the PSF convolution is already
    applied on the image.  This is useful primarily when you are more interested in the
    spectral properties (fluorophores, filters, bleedthrough, etc.) than the spatial
    properties (PSF, modality, etc.) in the simulation.
    """

    def render(
        self,
        truth: xrDataArray,  # (F, Z, Y, X)
        fluorescence: Sequence[Sequence[ChannelFluorescence]],  # [channel][fluor]
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
    ) -> xrDataArray:
        """Render a 3D image of the truth for F fluorophores, in C channels.

        In this case we don't apply the PSF convolution, as the truth is assumed to be
        already convolved with the PSF. Therefore, we simply compute the emission flux
        for each fluorophore and each channel.
        """
        fluors = truth.coords[Axis.F].values
        rates = []
        for channel in fluorescence:
            by_fluor = {fl.fluorophore: fl for fl in channel}
            rates.append([by_fluor[fluor].detected_rate() for fluor in fluors])
        em_rates = DataArray(
            xp.asarray(np.asarray(rates), dtype=truth.dtype),
            dims=[Axis.C, Axis.F],
            coords={Axis.C: [ch[0].channel for ch in fluorescence]},
        )
        em_image = em_rates * truth
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


def _detected_light_center(fluorescence: Sequence[ChannelFluorescence]) -> float:
    """Mean wavelength of all light detected (all channels and fluorophores given)."""
    total: float = 0
    weighted: float = 0
    for fl in fluorescence:
        detected = fl.detected_spectrum(saturate=False)
        total += float(detected.sum())
        weighted += float((detected * detected.coords[Axis.W]).sum())
    if total <= 0:
        names = sorted({fl.channel.name for fl in fluorescence})
        raise ValueError(
            "Cannot determine the pinhole size: no light is detected in channel(s) "
            f"{names}.  Set `Confocal.airy_unit_wavelength_nm`."
        )
    return weighted / total


def emission_lines(
    detection: xrDataArray, *, bins: int, threshold_percentage: float
) -> SpectralLines:
    """Discretize an emission spectrum into `bins` `(wavelength, weight)` pairs.

    Each bin is represented by its centroid wavelength, and weights sum to 1.
    """
    binned = bin_spectrum(
        detection, bins=bins, threshold_percentage=threshold_percentage
    )
    lines = [
        (float(wvl), float(weight))
        for wvl, weight in zip(binned.coords[Axis.W].values, binned.values, strict=True)
        if weight > 0 and not np.isnan(wvl)
    ]
    total = sum(weight for _, weight in lines)
    return tuple((_round_nm(wvl), float(f"{w / total:.4g}")) for wvl, w in lines)


def excitation_lines(
    absorption: xrDataArray, *, threshold_percentage: float = 1, bin_nm: float = 10
) -> SpectralLines:
    """Discretize an absorption spectrum into `(wavelength, weight)` pairs.

    Wavelengths absorbing more than `threshold_percentage` of the peak are grouped
    into bins at most `bin_nm` wide, each represented by its centroid wavelength.
    The weights are each bin's fraction of the total (thresholded) absorption.
    Laser lines more than `bin_nm` apart are never grouped together.
    """
    wvl = np.asarray(absorption.coords[Axis.W].values, dtype=float)
    rate = np.asarray(absorption.values, dtype=float)
    keep = rate > rate.max() * threshold_percentage / 100
    wvl, rate = wvl[keep], rate[keep]

    idx = ((wvl - wvl.min()) // bin_nm).astype(int)
    sums = np.bincount(idx, weights=rate)
    centroids = np.bincount(idx, weights=rate * wvl)[sums > 0] / sums[sums > 0]
    weights = sums[sums > 0] / sums.sum()
    return tuple(
        (_round_nm(w), float(f"{f:.4g}"))
        for w, f in zip(centroids, weights, strict=True)
    )


# PSF wavelengths are rounded to this step, so that fluorophores with nearly the
# same (centroid) wavelength share PSFs.  The PSF width scales with wavelength, so
# this changes it by < 0.1%.
_WAVELENGTH_STEP_NM = 0.5


def _round_nm(wvl_nm: float) -> float:
    return float(round(wvl_nm / _WAVELENGTH_STEP_NM) * _WAVELENGTH_STEP_NM)


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
    binned = masked.groupby_bins(Axis.W, bins=bins, include_lowest=True)

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
    return nx | 1  # odd, so that the PSF is centered on a pixel
