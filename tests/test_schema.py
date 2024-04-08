from collections.abc import Callable

import pytest

from microsim.schema import Simulation
from microsim.schema.channel import Channel
from microsim.schema.lens import ObjectiveLens
from microsim.schema.modality import Confocal, Modality, Widefield
from microsim.schema.samples import FluorophoreDistribution, MatsLines, Sample
from microsim.schema.settings import BackendName
from microsim.schema.space import ShapeScaleSpace

TRUTH_SPACE = ShapeScaleSpace(shape=(64, 128, 128), scale=(0.02, 0.01, 0.01))
MATSLINES = MatsLines(density=0.4, length=30, azimuth=5, max_r=1)
GREEN_MATSLINES = FluorophoreDistribution(distribution=MATSLINES)
NA1_4 = ObjectiveLens(numerical_aperture=1.4)
FITC = Channel(name="FITC", excitation=488, emission=525)
CONFOCAL_AU0_2 = Confocal(pinhole=0.2)
WIDEFIELD = Widefield()


@pytest.fixture
def sim1() -> Simulation:
    return Simulation(
        truth_space=TRUTH_SPACE,
        output_space={"downscale": 8},
        samples=[Sample(labels=[GREEN_MATSLINES])],
        objective_lens=NA1_4,
        channels=[FITC],
    )


@pytest.mark.parametrize("modality", [WIDEFIELD, CONFOCAL_AU0_2], ids=lambda x: x.type)
def test_schema(
    sim1: Simulation, np_backend: BackendName, modality: Modality, benchmark: Callable
) -> None:
    sim1.settings.np_backend = np_backend
    sim1.modality = modality

    output = benchmark(sim1.run)
    assert output.shape == sim1.output_space.shape
