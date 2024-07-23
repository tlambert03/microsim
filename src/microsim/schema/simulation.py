import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import AfterValidator, Field, model_validator

from microsim._data_array import ArrayProtocol, from_cache, to_cache
from microsim.interval_creation import Bin, bin_spectrum, generate_bins
from microsim.schema._emission import get_emission_events
from microsim.util import microsim_cache

from ._base_model import SimBaseModel
from .detectors import Detector
from .dimensions import Axis
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .optical_config import OpticalConfig
from .optical_config.lib import FITC
from .sample import FluorophoreDistribution, Sample
from .settings import Settings
from .space import ShapeScaleSpace, Space, _RelativeSpace
from .spectrum import Spectrum

if TYPE_CHECKING:
    from typing import Self, TypedDict, Unpack

    from .backend import NumpyAPI

    class SimluationKwargs(TypedDict, total=False):
        output_space: Space | dict | None
        objective_lens: ObjectiveLens
        channels: list[OpticalConfig]
        detector: Detector | None
        modality: Modality
        settings: Settings
        output_path: "OutPath" | None


def _check_extensions(path: Path) -> Path:
    if path.suffix not in {".tif", ".tiff", ".zarr", ".nc"}:
        raise ValueError("Recognized extensions include: .tif, .tiff, .zarr")
    return path


OutPath = Annotated[Path, AfterValidator(_check_extensions)]
xr.set_options(keep_attrs=True)  # type: ignore [no-untyped-call]


class Simulation(SimBaseModel):
    """Top level Simulation object."""

    truth_space: Space
    output_space: Space | None = None
    sample: Sample
    modality: Modality = Field(default_factory=Widefield)
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[OpticalConfig] = Field(default_factory=lambda: [FITC])
    # TODO: channels should also include `lights: list[LightSource]`
    detector: Detector | None = None
    settings: Settings = Field(default_factory=Settings)
    output_path: OutPath | None = None
    emission_bins: int = 3

    @classmethod
    def from_ground_truth(
        self,
        ground_truth: ArrayProtocol,
        scale: tuple[float, ...],
        **kwargs: "Unpack[SimluationKwargs]",
    ) -> "Self":
        """Shortcut to create a simulation directly from a ground truth array.

        In this case, we bypass derive the `truth_space` and `sample` objects directly
        from a pre-calculated ground truth array.  `scale` must also be provided as a
        tuple of floats, one for each dimension of the ground truth array.
        """
        return self(
            truth_space=ShapeScaleSpace(shape=ground_truth.shape, scale=scale),
            sample=Sample(labels=[FluorophoreDistribution.from_array(ground_truth)]),
            **kwargs,
        )

    @model_validator(mode="after")
    def _resolve_spaces(self) -> "Self":
        if isinstance(self.truth_space, _RelativeSpace):
            if self.output_space is not None:
                if isinstance(self.output_space, _RelativeSpace):
                    raise ValueError("Cannot have two relative spaces.")
                self.truth_space.reference = self.output_space
        elif isinstance(self.output_space, _RelativeSpace):
            self.output_space.reference = self.truth_space
        return self

    @property
    def _xp(self) -> "NumpyAPI":
        return self.settings.backend_module()

    def _wavelength_bins(self) -> list[Bin]:
        """Create wavelength bins depending on flurophores emission spectra.

        Bins placement is based on the fluorophores emission spectra in the ground
        truth sample. If fluorophores are not defined, bins are created in the range
        identified by [`settings.min_wavelength`-`settings.max_wavelength`].
        """
        return self._get_wavelength_bins()

    def run(self, channels: int | Sequence[int] | None = None) -> xr.DataArray:
        """Run the simulation and return the result.

        This will also write a file to disk if `output` is set.
        """
        truth = self.ground_truth()
        if channels is None:
            channels = tuple(range(len(self.channels)))
        elif isinstance(channels, int):
            channels = (channels,)
        images = []
        for channel_idx in channels:
            # TODO: implement emission_flux given new illumination implementation
            emission_flux = self.emission_flux(truth, channel_idx=channel_idx)
            optical_image = self.optical_image(emission_flux, channel_idx=channel_idx)
            images.append(self.digital_image(optical_image))
        image = xr.concat(images, dim=Axis.C)
        self._write(image)
        return image

    def ground_truth(self) -> xr.DataArray:
        """Return the ground truth data."""
        if not hasattr(self, "_ground_truth"):
            xp = self._xp
            # make empty space into which we'll add the ground truth
            # TODO: this is wasteful... label.render should probably
            # accept the space object directly
            truth = self.truth_space.create(array_creator=xp.zeros)

            # render each ground truth
            label_data = []
            for label in self.sample.labels:
                cache_path = self._truth_cache_path(
                    label, self.truth_space, self.settings.random_seed
                )
                if self.settings.cache.read and cache_path and cache_path.exists():
                    data = from_cache(cache_path, xp=xp).astype(
                        self.settings.float_dtype
                    )
                    logging.info(
                        f"Loaded ground truth for {label} from cache: {cache_path}"
                    )
                else:
                    data = label.render(truth, xp=xp)
                    if self.settings.cache.write and cache_path:
                        to_cache(data, cache_path, dtype=np.uint16)

                label_data.append(data)

            # concat along the F axis
            truth = xr.concat(label_data, dim=Axis.F)
            truth.coords.update({Axis.F: list(self.sample.labels)})
            truth.attrs.update(unit="counts")
            self._ground_truth = truth
        return self._ground_truth

    def _truth_cache_path(
        self,
        label: "FluorophoreDistribution",
        truth_space: "Space",
        seed: int | None,
    ) -> Path | None:
        if not (lbl_path := label.cache_path()):
            return None

        truth_cache = Path(microsim_cache("ground_truth"), *lbl_path)
        shape = f'shape{"_".join(str(x) for x in truth_space.shape)}'
        scale = f'scale{"_".join(str(x) for x in truth_space.scale)}'
        truth_cache = truth_cache / shape / scale
        if label.distribution.is_random():
            truth_cache = truth_cache / f"seed{seed}"
        return truth_cache

    def _get_wavelength_bins(self) -> list[Bin]:
        """Create wavelength bins depending on flurophores emission spectra.

        Bins placement is based on the fluorophores emission spectra in the ground
        truth sample. If fluorophores are not defined, bins are created in the range
        identified by [`settings.min_wavelength`-`settings.max_wavelength`].

        NOTE: We assume binning to be independent of the excitation/emission filters
        applied, so we only consider the emission spectra of the fluorophores.
        Why? Because the excitation filters are applied to the light source, and
        emission filters are after illumination of fluorophores, for which we need
        matching bins.
        """
        fluorophores = [x.fluorophore for x in self.sample.labels]
        if len(fluorophores) == 0:
            # we have no fluorophores to calculate
            raise ValueError("No fluorophores in the current sample!")

        # Get emission spectra for all the fluorophores
        fluor_em_spectra = []
        for fluor in fluorophores:
            if fluor is None:
                fluor_em_spectra.append(
                    np.array(
                        [self.settings.min_wavelength, self.settings.max_wavelength]
                    )
                )
            else:
                # get emission Spectrum for the given fluorophore
                fluor_em_spectra.append(fluor.emission_spectrum.wavelength.magnitude)

        # Get the min and max wavelength over all the spectra
        min_wave = min([x.min() for x in fluor_em_spectra])
        max_wave = max([x.max() for x in fluor_em_spectra])

        # Create the same bins for all the spectra
        wave_range = np.arange(min_wave, max_wave, 1)
        em_bins = generate_bins(
            x=wave_range,
            y=None,
            num_bins=self.settings.num_wavelength_bins,
            strategy=self.settings.binning_strategy,
        )
        return em_bins

    def illumination_flux(
        self,
        truth: xr.DataArray | None = None,
        channel_idx: int = 0,
        *,
        light_power: float = 100,
    ) -> xr.DataArray:
        """
        Return the illumination data as an array of shape (W, C, Z, Y, X).

        NOTE: we assume this happens before the excitation filters are applied.

        NOTE: for the moment we assume the light source to be the same over all
        the spatial dimensions. Only dimension is the wavelength.
        """
        if truth is None:
            truth = self.ground_truth()
        elif not isinstance(truth, xr.DataArray):
            raise ValueError("truth must be a DataArray")

        channel = self.channels[channel_idx]
        illum = channel.illumination
        if not illum:
            # If illumination is not defined, we assume a white light source
            illum = Spectrum(
                wavelength=np.arange(
                    self.settings.min_wavelength, self.settings.max_wavelength, 1
                ),
                intensity=np.ones(
                    self.settings.max_wavelength - self.settings.min_wavelength
                )
                * light_power,
            )
        # Bin illum spectrum
        binned_illum = bin_spectrum(spectrum=illum, bins=self._wavelength_bins())  # (W)
        # Broadcast to (W, Z, Y, X)
        binned_illum = binned_illum.expand_dims(
            [Axis.Z, Axis.Y, Axis.X], axis=[1, 2, 3]
        )
        spatial_illum = binned_illum * np.ones((1, *truth.shape[1:]))  # (W, Z, Y, X)
        spatial_illum = spatial_illum.expand_dims([Axis.C], axis=1)  # (W, 1, Z, Y, X)
        spatial_illum.coords.update(
            {
                Axis.C: [channel],
                Axis.Z: truth.coords[Axis.Z],
                Axis.Y: truth.coords[Axis.Y],
                Axis.X: truth.coords[Axis.X],
            }
        )
        return spatial_illum

    def emission_flux(
        self, truth: "xr.DataArray | None" = None, *, channel_idx: int = 0
    ) -> xr.DataArray:
        if truth is None:
            truth = self.ground_truth()
        elif not isinstance(truth, xr.DataArray):
            raise ValueError("truth must be a DataArray")

        if Axis.F not in truth.coords:
            # we have no fluorophores to calculate
            return truth

        channel = self.channels[channel_idx]  # TODO
        emission_flux_arr = []
        for f_idx, fluor_dist in enumerate(truth.coords[Axis.F].values):
            fluor = cast(FluorophoreDistribution, fluor_dist).fluorophore
            fluor_counts = truth[{Axis.F: f_idx}]
            fluor_counts = fluor_counts.expand_dims(
                [Axis.W, Axis.C, Axis.F], axis=[0, 1, 2]
            )
            if fluor is None:
                # TODO
                # what here?  should we pick a default fluor?
                default_bin = [
                    pd.Interval(
                        left=self.settings.min_wavelength,
                        right=self.settings.max_wavelength,
                    )
                ]
                fluor_counts = fluor_counts.assign_coords(w=default_bin)
                emission_flux_arr.append(fluor_counts)
            else:
                em_spectrum = get_emission_events(channel, fluor)
                binned_events = bin_spectrum(
                    spectrum=em_spectrum,
                    bins=None,  # TODO: use the same bins as illumination?
                    num_bins=self.emission_bins,  # TODO: same num_bins as illumination?
                    binning_strategy="equal_area",  # to be consistent with PR#35
                )
                # TODO: This is not stochastic.
                # every pixel ideally could have a different binned_events.

                fluor_counts = xr.concat(
                    [fluor_counts * x.values.item() for x in binned_events],
                    dim=Axis.W,
                )
                fluor_counts = fluor_counts.assign_coords(
                    w=binned_events[Axis.W].values
                )
            # (W, C, F, Z, Y, X)
            emission_flux_arr.append(fluor_counts)

        emission_flux_data = xr.concat(emission_flux_arr, dim=Axis.F)
        emission_flux_data.attrs.update(unit="photon/sec")
        return emission_flux_data

    def optical_image(
        self, emission_flux: xr.DataArray | None = None, *, channel_idx: int = 0
    ) -> xr.DataArray:
        if emission_flux is None:
            emission_flux = self.emission_flux()
        # Input has the following co-ordinates: (W, C, F, Z, Y, X)
        # let the given modality render the as an image (convolved, etc..)
        result = self.modality.render(
            emission_flux,
            self.channels[channel_idx],
            objective_lens=self.objective_lens,
            settings=self.settings,
            xp=self._xp,
        )

        # Co-ordinates: (C, Z, Y, X)
        return result

    def digital_image(
        self,
        optical_image: xr.DataArray | None = None,
        *,
        photons_pp_ps_max: int = 10000,
        exposure_ms: float = 100,
        with_detector_noise: bool = True,
        channel_idx: int = 0,
    ) -> xr.DataArray:
        if optical_image is None:
            optical_image = self.optical_image(channel_idx=channel_idx)
        image = optical_image
        # TODO:Multi-fluorophore setup: combine information present in all wavelength
        # intervals.
        if self.output_space is not None:
            image = self.output_space.rescale(image)
        if self.detector is not None and with_detector_noise:
            im_max = self._xp.max(image.data)
            if not np.any(im_max):
                photon_flux = image
            else:
                photon_flux = image * photons_pp_ps_max / im_max
            gray_values = self.detector.render(
                photon_flux, exposure_ms=exposure_ms, xp=self._xp
            )
            image = gray_values
        image.attrs.update(unit="gray values")
        return image

    def _write(self, result: xr.DataArray) -> None:
        if not self.output_path:
            return
        if hasattr(result.data, "get"):
            result = result.copy(data=result.data.get(), deep=False)
        result.attrs["microsim.Simulation"] = self.model_dump_json()
        result.attrs.pop("space", None)
        result.coords[Axis.C] = [c.name for c in result.coords[Axis.C].values]
        if self.output_path.suffix == ".zarr":
            result.to_zarr(self.output_path, mode="w")
        elif self.output_path.suffix in (".nc",):
            result.to_netcdf(self.output_path)
        elif self.output_path.suffix in (".tif", ".tiff"):
            import tifffile as tf

            tf.imwrite(self.output_path, np.asanyarray(result))
