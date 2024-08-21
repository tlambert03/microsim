import logging
import warnings
from typing import Annotated, Any, Literal

import numpy as np
from annotated_types import Ge

from microsim._data_array import ArrayProtocol, DataArray, xrDataArray
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.dimensions import Axis
from microsim.schema.lens import ObjectiveLens
from microsim.schema.settings import Settings
from microsim.schema.space import SpaceProtocol


class _PSFModality(SimBaseModel):
    def psf(
        self,
        space: SpaceProtocol,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
        ex_wvl_nm: float | None = None,
        em_wvl_nm: float | None = None,
    ) -> ArrayProtocol:
        # default implementation is a widefield PSF
        nz, _ny, nx = space.shape
        dz, _dy, dx = space.scale
        return make_psf(
            nz=nz,
            nx=nx,
            dx=dx,
            dz=dz,
            objective=objective_lens,
            ex_wvl_nm=ex_wvl_nm,
            em_wvl_nm=em_wvl_nm,
            max_au_relative=settings.max_psf_radius_aus,
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
            logging.info(f"Rendering channel {ch} -----------------")

            # for every fluorophore in the sample...
            fluors = []
            for fluor in em_rates.coords[Axis.F].values:
                logging.info(f">> fluor {fluor}")
                f_truth = truth.sel({Axis.F: fluor})

                # discretize the emission spectrum for this specific ch/fluor pair
                em_spectrum = em_rates.sel({Axis.C: ch, Axis.F: fluor})
                if not (em_spectrum > 1e-12).any():
                    # no emission at all for this fluorophore in this channel
                    fluors.append(xp.zeros_like(f_truth))
                    continue

                binned = bin_spectrum(
                    em_spectrum,
                    bins=settings.spectral_bins_per_emission_channel,
                    threshold_percentage=settings.spectral_bin_threshold_percentage,
                )

                # Create a full PSF-convoled image for each emission wavelength bin
                # and sum them together.  (More bins create a more realistic
                # superposition of wavelength-specific PSFs, at the cost of time).
                fluor_sum: Any = 0
                for em_rate, em_wvl_nm in zip(
                    binned, binned[Axis.W].values, strict=True
                ):
                    if xp.isnan(em_rate) or em_rate == 0 or xp.isnan(em_wvl_nm):
                        continue
                    logging.info(f">>>> @ {em_wvl_nm} nm")

                    # multiply the truth (fluorophore distribution) by the emission rate
                    # this gives us a (Z, Y, X) array of photons/sec
                    binned_flux = f_truth * em_rate
                    # create the PSF for this emission wavelength
                    psf = self.psf(
                        truth.attrs["space"],
                        objective_lens=objective_lens,
                        em_wvl_nm=em_wvl_nm,
                        settings=settings,
                        xp=xp,
                    )
                    fluor_sum += xp.fftconvolve(binned_flux, psf, mode="same")
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


class Confocal(_PSFModality):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

    def psf(
        self,
        space: SpaceProtocol,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
        ex_wvl_nm: float | None = None,
        em_wvl_nm: float | None = None,
    ) -> ArrayProtocol:
        nz, _ny, nx = space.shape
        dz, _dy, dx = space.scale
        return make_psf(
            nz=nz,
            nx=nx,
            dx=dx,
            dz=dz,
            objective=objective_lens,
            em_wvl_nm=em_wvl_nm,
            ex_wvl_nm=ex_wvl_nm,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )


class Widefield(_PSFModality):
    type: Literal["widefield"] = "widefield"


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
        w_min, w_max = masked.w.min(), masked.w.max()
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
    try:
        binned = masked.groupby_bins(Axis.W, bins=bins)
    except ValueError as e:
        breakpoint()

    # Create a new DataArray with the summed intensities and centroid wavelengths
    binned_spectrum = binned.sum(Axis.W)

    # Calculate the centroid wavelength for each bin
    centroids = binned.map(lambda x: (x.w * x).sum() / x.sum())
    binned_spectrum.coords.update({Axis.W: centroids})
    # Swap the dimensions to make the wavelength centroid the primary dimension
    binned_spectrum = binned_spectrum.swap_dims({"w_bins": Axis.W})
    return binned_spectrum
