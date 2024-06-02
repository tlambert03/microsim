from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from .optical_config.lib import FITC

if TYPE_CHECKING:
    from typing_extensions import Self

    from .backend import NumpyAPI

from pydantic import AfterValidator, Field, model_validator

from microsim._data_array import ArrayProtocol, DataArray
from microsim.emission_bins import EmissionBins
from microsim.interval_creation import WavelengthSpace

from ._base_model import SimBaseModel
from .detectors import Detector
from .emission import get_emission_events
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .optical_config import OpticalConfig
from .sample import FluorophoreDistribution, Sample
from .settings import Settings
from .space import ShapeScaleSpace, Space, _RelativeSpace

if TYPE_CHECKING:
    from typing import TypedDict

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


class Simulation(SimBaseModel):
    """Top level Simulation object."""

    truth_space: Space
    output_space: Space | None = None
    sample: Sample
    modality: Modality = Field(default_factory=Widefield)
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[OpticalConfig] = Field(default_factory=lambda: [FITC])
    detector: Detector | None = None
    settings: Settings = Field(default_factory=Settings)
    output_path: OutPath | None = None
    emission_bins: int = 3

    @classmethod
    def from_ground_truth(
        self,
        ground_truth: ArrayProtocol,
        scale: tuple[float, ...],
        **kwargs: "Unpack[SimluationKwargs]",  # type: ignore # noqa: F821
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

    def run(self, channel_idx: int = 0) -> "DataArray":
        """Run the simulation and return the result.

        This will also write a file to disk if `output` is set.
        """
        truth = self.ground_truth()
        optical_image = self.optical_image(truth, channel_idx=channel_idx)
        image = self.digital_image(optical_image)
        self._write(image)
        return image

    def ground_truth(self) -> "DataArray":
        """Return the ground truth data."""
        if not hasattr(self, "_ground_truth"):
            xp = self._xp
            # make empty space into which we'll add fluorescence
            truth = self.truth_space.create(array_creator=xp.zeros)
            # add fluorophores to the space
            for label in self.sample.labels:
                truth = label.render(truth, xp=xp)
            truth.attrs["space"] = self.truth_space  # TODO
            self._ground_truth = truth
        return self._ground_truth

    # TODO: Multi-fluorophore setup: incident light spectrum needs to be passed to
    # the optical system as an additional arguement.
    def optical_image(
        self, truth: "DataArray | None" = None, *, channel_idx: int = 0
    ) -> "DataArray":
        if truth is None:
            truth = self.ground_truth()
        elif not isinstance(truth, DataArray):
            raise ValueError("truth must be a DataArray")
        # let the given modality render the as an image (convolved, etc..)
        channel = self.channels[channel_idx]  # TODO

        # get the emission events for the given fluorophore
        fluorophore_str = "EGFP"
        em_wavelengths, em_events = get_emission_events(
            "wKqWb", "Widefield Green", fluorophore_str
        )
        ex_filter_str = f"{channel.excitation.bandcenter.magnitude}-\
            {channel.excitation.bandwidth.magnitude}"
        binned_events, wavelength_bins = EmissionBins.bin_events(
            fluorophore_str,
            ex_filter_str,
            self.emission_bins,
            em_wavelengths,
            em_events,
        )
        # import pdb; pdb.set_trace()

        # TODO: This is not stochastic. every pixel ideally could have a different binned_events.
        emitted_data = truth.data[None]
        emitted_data = self._xp.concatenate(
            [emitted_data * x.magnitude for x in binned_events], axis=0
        )
        # create a truth space with the binnded wavelengths
        truth = WavelengthSpace(
            wavelength_bins=wavelength_bins,
            data=emitted_data,
            space=truth.attrs["space"],
            coords=truth.coords,
        )
        # emitted_data= WavelengthSpace(wavelengths=binned_wavelengths, data=emitted_data)
        # Allocate the emitted light in wavelength intervals. (Pre-computed for
        #           each fluorophore)
        # For all of this, adapt the code from  examples/emission_events.py.
        result = self.modality.render(
            truth,
            channel,
            objective_lens=self.objective_lens,
            settings=self.settings,
            xp=self._xp,
        )

        return result

    def digital_image(
        self,
        optical_image: "DataArray | None" = None,
        *,
        photons_pp_ps_max: int = 2000,
        exposure_ms: float = 100,
        with_detector_noise: bool = True,
        channel_idx: int = 0,
    ) -> "DataArray":
        if optical_image is None:
            optical_image = self.optical_image(channel_idx=channel_idx)
        image = optical_image
        # TODO:Multi-fluorophore setup: combine information present in all wavelength
        # intervals.
        if self.output_space is not None:
            image = self.output_space.rescale(image)
        if self.detector is not None and with_detector_noise:
            photon_flux = image.data * photons_pp_ps_max / self._xp.max(image.data)
            gray_values = self.detector.render(
                photon_flux, exposure_ms=exposure_ms, xp=self._xp
            )
            image = DataArray(gray_values, coords=image.coords, attrs=image.attrs)
        return image

    def _write(self, result: "DataArray") -> None:
        if not self.output_path:
            return
        self_json = self.model_dump_json()
        if self.output_path.suffix == ".zarr":
            result.to_zarr(self.output_path, mode="w", attrs={"microsim": self_json})
        if self.output_path.suffix in (".tif", ".tiff"):
            result.to_tiff(self.output_path, description=self_json)
        if self.output_path.suffix in (".nc",):
            result.to_netcdf(self.output_path, attrs={"microsim": self_json})
