from pydantic import BaseModel, Field

from .channel import Channel
from .lens import ObjectiveLens
from .modality import Modality
from .sample import Sample
from .space import Space


class Simulation(BaseModel):
    truth_space: Space
    output_space: Space
    samples: list[Sample]
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[Channel]
    modality: Modality

    def run(self, sample_idx: int = 0, channel_idx: int = 0) -> None:
        from microsim.util import downsample, uniformly_spaced_xarray

        truth = uniformly_spaced_xarray(
            shape=self.truth_space.shape, scale=self.truth_space.scale
        )
        truth = self.samples[sample_idx].render(truth)
        truth.attrs["space"] = self.truth_space
        channel = self.channels[channel_idx]
        img = self.modality.render(truth, channel, self.objective_lens)
        downsample(img, self.output_space.downscale)
