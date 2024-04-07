from pydantic import BaseModel, Field, model_validator

from .channel import Channel
from .lens import ObjectiveLens
from .modality import Modality
from .sample import Sample
from .space import Space, _RelativeSpace


class Simulation(BaseModel):
    truth_space: Space
    output_space: Space
    samples: list[Sample]
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[Channel]
    modality: Modality

    @model_validator(mode="after")
    def _resolve_spaces(cls, value: "Simulation") -> "Simulation":
        if isinstance(value.truth_space, _RelativeSpace):
            if isinstance(value.output_space, _RelativeSpace):
                raise ValueError("Cannot have two relative spaces.")
            value.truth_space.reference = value.output_space
        elif isinstance(value.output_space, _RelativeSpace):
            value.output_space.reference = value.truth_space

    def run(self, sample_idx: int = 0, channel_idx: int = 0) -> None:
        from microsim.util import uniformly_spaced_xarray

        truth = uniformly_spaced_xarray(
            shape=self.truth_space.shape, scale=self.truth_space.scale
        )
        truth = self.samples[sample_idx].render(truth)
        truth.attrs["space"] = self.truth_space
        channel = self.channels[channel_idx]
        img = self.modality.render(truth, channel, self.objective_lens)
        img = self.output_space.rescale(img)
        return img
