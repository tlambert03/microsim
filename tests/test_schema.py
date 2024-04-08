from collections.abc import Callable
from pathlib import Path

import pytest

from microsim.schema import Simulation
from microsim.schema.backend import BackendName
from microsim.schema.channel import Channel
from microsim.schema.lens import ObjectiveLens
from microsim.schema.modality import Confocal, Modality, Widefield
from microsim.schema.samples import FluorophoreDistribution, MatsLines, Sample
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
        sample=Sample(labels=[GREEN_MATSLINES]),
        objective_lens=NA1_4,
        channels=[FITC],
    )


@pytest.mark.parametrize("modality", [WIDEFIELD, CONFOCAL_AU0_2], ids=lambda x: x.type)
def test_schema(
    sim1: Simulation,
    np_backend: BackendName,
    modality: Modality,
    benchmark: Callable,
    tmp_path: Path,
) -> None:
    sim1.settings.np_backend = np_backend
    sim1.modality = modality
    sim1.output = tmp_path / "output.zarr"

    output = benchmark(sim1.run)
    assert output.shape == sim1.output_space.shape


@pytest.mark.parametrize("ext", [".tif", ".zarr", ".nc"])
def test_simulation_output(tmp_path: Path, ext: str) -> None:
    sim = Simulation(
        truth_space=ShapeScaleSpace(shape=(64, 128, 128), scale=(0.2, 0.1, 0.1)),
        output_space={"downscale": 1},
        sample=Sample(labels=[GREEN_MATSLINES]),
        objective_lens=NA1_4,
        channels=[FITC],
        output=tmp_path / f"output.{ext}",
    )
    sim.run()
