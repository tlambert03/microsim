from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from typing import Self

from pydantic import AfterValidator, BaseModel, Field, model_validator

from microsim._data_array import DataArray

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
    output_space: Space | None = None
    sample: Sample
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[Channel] = Field(
        default_factory=lambda: [Channel(name="FITC", excitation=488, emission=525)]
    )
    modality: Modality = Field(default_factory=Widefield)
    settings: Settings = Field(default_factory=Settings)
    output: OutPath | None = None

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

    def run(self, channel_idx: int = 0) -> DataArray:
        xp = self.settings.backend_module()
        channel = self.channels[channel_idx]

        # make empty space into which we'll add fluorescence
        truth = self.truth_space.create(array_creator=xp.zeros)

        # add fluorophores to the space
        for label in self.sample.labels:
            truth = label.render(truth, xp=xp)
        truth.attrs["space"] = self.truth_space  # TODO

        # let the given modality render the as an image (convolved, etc..)
        result = self.modality.render(truth, channel, self.objective_lens, xp=xp)
        if self.output_space is not None:
            result = self.output_space.rescale(result)
        self._write(result)
        return result

    def _write(self, result: DataArray) -> None:
        if not self.output:
            return
        self_json = self.model_dump_json()
        if self.output.suffix == ".zarr":
            result.to_zarr(self.output, mode="w", attrs={"microsim": self_json})
        if self.output.suffix in (".tif", ".tiff"):
            result.to_tiff(self.output, description=self_json)
        if self.output.suffix in (".nc",):
            result.to_netcdf(self.output, attrs={"microsim": self_json})
