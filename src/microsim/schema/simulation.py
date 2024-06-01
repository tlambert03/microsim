import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from typing import Self

    from microsim.schema.sample.sample import FluorophoreDistribution

    from .backend import NumpyAPI

import numpy as np
from pydantic import AfterValidator, Field, model_validator

from microsim._data_array import DataArray

from ._base_model import SimBaseModel
from .detectors import Detector
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .optical_config import FITC, OpticalConfig
from .sample import Sample
from .settings import Settings
from .space import Space, _RelativeSpace


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
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[OpticalConfig] = Field(default_factory=lambda: [FITC()])
    detector: Detector | None = None
    modality: Modality = Field(default_factory=Widefield)
    settings: Settings = Field(default_factory=Settings)
    output_path: OutPath | None = None

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
        print("starting ground truth")
        if not hasattr(self, "_ground_truth"):
            xp = self._xp
            # make empty space into which we'll add fluorescence
            print("creating truth space")
            # FIXME ... this is slower than it needs to be (creating all the zeros)
            truth = self.truth_space.create(array_creator=xp.zeros)
            truth.attrs["space"] = self.truth_space  # TODO, hack

            # hack... we're going to/from xarray here
            xt = truth.to_xarray().copy()
            xt = xt.expand_dims(dim={"L": len(self.sample.labels)}).copy()

            # add fluorophores to the space
            for n, label in enumerate(self.sample.labels):
                cache_path = self._truth_cache_path(
                    label, self.truth_space, self.settings.random_seed
                )
                if cache_path and cache_path.exists():
                    lbl_data = DataArray.from_cache(cache_path)
                    logging.info(f"Loaded ground truth from cache: {cache_path}")
                else:
                    lbl_data = label.render(truth, xp=xp)
                    if cache_path:
                        lbl_data.to_cache(cache_path, dtype=np.uint16)
                xt[{"L": n}] = lbl_data
            self._ground_truth = DataArray.from_xarray(xt)
        return self._ground_truth

    def _truth_cache_path(
        self,
        label: "FluorophoreDistribution",
        truth_space: "Space",
        seed: int | None,
    ) -> Path | None:
        from microsim.util import MICROSIM_CACHE

        if not (lbl_path := label.cache_path()):
            return None
        cache_path = Path(MICROSIM_CACHE, "truth_cache", *lbl_path)

        shape = f'shape{"_".join(str(x) for x in truth_space.shape)}'
        scale = f'scale{"_".join(str(x) for x in truth_space.scale)}'
        cache_path = cache_path / shape / scale
        return cache_path

    def optical_image(
        self, truth: "DataArray | None" = None, *, channel_idx: int = 0
    ) -> "DataArray":
        if truth is None:
            truth = self.ground_truth()
        elif not isinstance(truth, DataArray):
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

        return result

    def digital_image(
        self,
        optical_image: "DataArray | None" = None,
        *,
        photons_pp_ps_max: int = 10000,
        exposure_ms: float = 100,
        with_detector_noise: bool = True,
        channel_idx: int = 0,
    ) -> "DataArray":
        if optical_image is None:
            optical_image = self.optical_image(channel_idx=channel_idx)
        image = optical_image
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
