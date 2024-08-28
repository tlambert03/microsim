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
        *,
        nz: int,
        nx: int,
        dx: float,
        dz: float,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI,
        ex_wvl_nm: float | None = None,
        em_wvl_nm: float | None = None,
    ) -> ArrayProtocol:
        # default implementation is a widefield PSF
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
            logging.info(f"Rendering channel {ch} -----------------")

            # for every fluorophore in the sample...
            fluors = []
            for f_idx, fluor in enumerate(truth.coords[Axis.F].values):
                logging.info(f">> fluor {fluor}")
                f_truth = truth.isel({Axis.F: f_idx})

                # discretize the emission spectrum for this specific ch/fluor pair
                em_spectrum = em_rates.sel({Axis.C: ch, Axis.F: fluor})
                # if we happen to have 2 spectra for the same fluorophore
                # in the same channel, just take the first one (shouldn't happen)
                if Axis.F in em_spectrum.dims:  # pragma: no cover
                    em_spectrum = em_spectrum.isel({Axis.F: 0})

                if not (em_spectrum > 1e-12).any():
                    # no emission at all for this fluorophore in this channel
                    fluors.append(xp.zeros_like(f_truth))
                    continue

                summed_psf = self._summed_weighted_psf(
                    em_spectrum, settings, truth.attrs["space"], objective_lens, xp
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

        summed_psf: Any = 0
        for em_rate in binned:
            em_wvl_nm = em_rate.w.item()
            if em_rate.isnull().any() or em_rate == 0 or xp.isnan(em_wvl_nm):
                continue
            weight = em_rate.item()
            logging.info(f">>>> PSF @ {em_wvl_nm:.1f}nm (x{weight:.2f})")
            psf = self.psf(
                nz=nz,
                nx=nx,
                dx=dx,
                dz=dz,
                objective_lens=objective_lens,
                em_wvl_nm=em_wvl_nm,
                xp=xp,
            )
            summed_psf += psf * weight
        return summed_psf  # type: ignore [no-any-return]


class Confocal(_PSFModality):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

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
