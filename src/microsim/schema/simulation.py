from pathlib import Path
from typing import Annotated, Self

import xarray as xr
from pydantic import AfterValidator, BaseModel, Field, model_validator

from .channel import Channel
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .samples import Sample
from .settings import Settings
from .space import Space, _RelativeSpace


def _check_extensions(path: Path) -> Path:
    if path.suffix not in {".tif", ".tiff", ".zarr", ".nc"}:
        raise ValueError("Recognized extensions include: .tif, .tiff, .zarr")
    return path


OutPath = Annotated[Path, AfterValidator(_check_extensions)]


class Simulation(BaseModel):
    truth_space: Space
    output_space: Space
    sample: Sample
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[Channel]
    modality: Modality = Field(default_factory=Widefield)
    settings: Settings = Field(default_factory=Settings)
    output: OutPath | None = None

    @model_validator(mode="after")
    def _resolve_spaces(self) -> Self:
        if isinstance(self.truth_space, _RelativeSpace):
            if isinstance(self.output_space, _RelativeSpace):
                raise ValueError("Cannot have two relative spaces.")
            self.truth_space.reference = self.output_space
        elif isinstance(self.output_space, _RelativeSpace):
            self.output_space.reference = self.truth_space
        return self

    def run(self, channel_idx: int = 0) -> xr.DataArray:
        xp = self.settings.backend_module()
        channel = self.channels[channel_idx]

        truth = self.truth_space.create(array_creator=xp.zeros)  # type: ignore
        for label in self.sample.labels:
            truth = label.render(truth, xp=xp)
        truth.attrs["space"] = self.truth_space  # TODO

        img = self.modality.render(truth, channel, self.objective_lens, xp=xp)
        result = self.output_space.rescale(img)
        self._write(result)
        return result

    def _write(self, result: xr.DataArray) -> None:
        if not self.output:
            return
        if self.output.suffix == ".zarr":
            result.to_zarr(self.output, mode="w")
        if self.output.suffix in (".tif", ".tiff"):
            import tifffile as tf

            desc = self.model_dump_json()
            tf.imwrite(self.output, result, description=desc)
        if self.output.suffix in (".nc",):
            result.to_netcdf(self.output)
