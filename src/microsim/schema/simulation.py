from typing import Self

import xarray as xr
from pydantic import BaseModel, Field, model_validator

from .channel import Channel
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .samples import Sample
from .settings import Settings
from .space import Space, _RelativeSpace


class Simulation(BaseModel):
    truth_space: Space
    output_space: Space
    samples: list[Sample]
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[Channel]
    modality: Modality = Field(default_factory=Widefield)
    settings: Settings = Field(default_factory=Settings)

    @model_validator(mode="after")
    def _resolve_spaces(self) -> Self:
        if isinstance(self.truth_space, _RelativeSpace):
            if isinstance(self.output_space, _RelativeSpace):
                raise ValueError("Cannot have two relative spaces.")
            self.truth_space.reference = self.output_space
        elif isinstance(self.output_space, _RelativeSpace):
            self.output_space.reference = self.truth_space
        return self

    def run(self, sample_idx: int = 0, channel_idx: int = 0) -> xr.DataArray:
        xp = self.settings.backend_module()
        sample = self.samples[sample_idx]
        channel = self.channels[channel_idx]

        truth = self.truth_space.create(array_creator=xp.zeros)  # type: ignore
        for label in sample.labels:
            truth = label.render(truth, xp=xp)
        truth.attrs["space"] = self.truth_space  # TODO

        img = self.modality.render(truth, channel, self.objective_lens, xp=xp)
        return self.output_space.rescale(img)
