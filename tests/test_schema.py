from pathlib import Path

import numpy as np
import pytest

import microsim.schema as ms

TRUTH_SPACE = ms.ShapeScaleSpace(shape=(64, 128, 128), scale=(0.04, 0.02, 0.02))
MATSLINES = ms.MatsLines(density=0.4, length=30, azimuth=5, max_r=1)
GREEN_MATSLINES = ms.FluorophoreDistribution(distribution=MATSLINES)
NA1_4 = ms.ObjectiveLens(numerical_aperture=1.4)
FITC = ms.Channel(name="FITC", excitation=488, emission=525)
CONFOCAL_AU0_2 = ms.Confocal(pinhole_au=0.2)
WIDEFIELD = ms.Widefield()


@pytest.fixture
def sim1() -> ms.Simulation:
    return ms.Simulation(
        truth_space=TRUTH_SPACE,
        output_space={"downscale": 4},
        sample=ms.Sample(labels=[GREEN_MATSLINES]),
        objective_lens=NA1_4,
        channels=[FITC],
    )


@pytest.mark.parametrize("seed", [None, 100])
@pytest.mark.parametrize("modality", [WIDEFIELD, CONFOCAL_AU0_2], ids=lambda x: x.type)
def test_schema(
    sim1: ms.Simulation,
    np_backend: ms.BackendName,
    modality: ms.Modality,
    tmp_path: Path,
    seed: int | None,
) -> None:
    sim1.settings.np_backend = np_backend
    sim1.modality = modality
    sim1.output = tmp_path / "output.zarr"
    sim1.settings.random_seed = seed

    out1 = sim1.run()
    assert out1.shape == sim1.output_space.shape

    # ensure we have the right datatype
    # this is tough with xarray proper... so we use our own DataArray wrapper.
    # https://github.com/google/jax/issues/17107
    # https://github.com/pydata/xarray/issues/7848
    assert type(out1.data).__module__.split(".")[0].startswith(np_backend)

    out2 = sim1.run()
    if seed is None:
        assert not np.allclose(out1, out2)
    else:
        np.testing.assert_allclose(out1, out2)


@pytest.mark.parametrize("ext", [".tif", ".zarr", ".nc"])
def test_simulation_output(tmp_path: Path, ext: str) -> None:
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(64, 128, 128), scale=(0.2, 0.1, 0.1)),
        output_space={"downscale": 1},
        sample=ms.Sample(labels=[GREEN_MATSLINES]),
        objective_lens=NA1_4,
        channels=[FITC],
        output=tmp_path / f"output.{ext}",
    )
    sim.run()
