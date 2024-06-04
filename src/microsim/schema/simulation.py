from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np

from .optical_config.lib import FITC

if TYPE_CHECKING:
    from typing import Self, Unpack

    from .backend import NumpyAPI

import xarray as xr
from pydantic import AfterValidator, Field, model_validator

from microsim._data_array import ArrayProtocol

from ._base_model import SimBaseModel
from .detectors import Detector
from .dimensions import Axis
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
xr.set_options(keep_attrs=True)


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

    def run(self, channel_idx: int = 0) -> xr.DataArray:
        """Run the simulation and return the result.

        This will also write a file to disk if `output` is set.
        """
        truth = self.ground_truth()
        optical_image = self.optical_image(truth, channel_idx=channel_idx)
        image = self.digital_image(optical_image)
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
            label_data = [label.render(truth, xp=xp) for label in self.sample.labels]
            # concat along the F axis
            truth = xr.concat(label_data, dim=Axis.F)
            truth.coords.update({Axis.F: list(self.sample.labels)})
            self._ground_truth = truth
        return self._ground_truth

    def optical_image(
        self, truth: "xr.DataArray | None" = None, *, channel_idx: int = 0
    ) -> xr.DataArray:
        if truth is None:
            truth = self.ground_truth()
        elif not isinstance(truth, xr.DataArray):
            raise ValueError("truth must be a DataArray")
        # let the given modality render the as an image (convolved, etc..)
        channel = self.channels[channel_idx]  # TODO
        result = self.modality.render(
            truth,
            channel,
            objective_lens=self.objective_lens,
            settings=self.settings,
            xp=self._xp,
        )

        # TODO: this is an oversimplification
        # it works for now, since we only have 1 Fluor and 1 Channel...
        # but there will not necessarily be a 1-to-1 mapping between F and C
        result = result.rename({Axis.F: Axis.C}).assign_coords({Axis.C: [channel]})
        return result

    def digital_image(
        self,
        optical_image: "xr.DataArray | None" = None,
        *,
        photons_pp_ps_max: int = 2000,
        exposure_ms: float = 100,
        with_detector_noise: bool = True,
        channel_idx: int = 0,
    ) -> xr.DataArray:
        if optical_image is None:
            optical_image = self.optical_image(channel_idx=channel_idx)
        image = optical_image
        if self.output_space is not None:
            image = self.output_space.rescale(image)
        if self.detector is not None and with_detector_noise:
            photon_flux = image * photons_pp_ps_max / self._xp.max(image.data)
            gray_values = self.detector.render(
                photon_flux, exposure_ms=exposure_ms, xp=self._xp
            )
            image = gray_values
        return image

    def _write(self, result: xr.DataArray) -> None:
        if not self.output_path:
            return
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
