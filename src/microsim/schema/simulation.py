from pydantic import BaseModel, Field, model_validator

from .channel import Channel
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .sample import Sample
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
    def _resolve_spaces(cls, value: "Simulation") -> "Simulation":
        if isinstance(value.truth_space, _RelativeSpace):
            if isinstance(value.output_space, _RelativeSpace):
                raise ValueError("Cannot have two relative spaces.")
            value.truth_space.reference = value.output_space
        elif isinstance(value.output_space, _RelativeSpace):
            value.output_space.reference = value.truth_space

    def run(self, sample_idx: int = 0, channel_idx: int = 0) -> None:
        xp = self.settings.backend_module()
        sample = self.samples[sample_idx]
        channel = self.channels[channel_idx]

        space = self.truth_space.create(xp.zeros)
        truth = sample.render(space, xp=xp)
        truth.attrs["space"] = self.truth_space  # TODO

        img = self.modality.render(truth, channel, self.objective_lens, xp=xp)
        img = self.output_space.rescale(img)
        return img
