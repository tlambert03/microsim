import pickle
from pathlib import Path

import numpy as np
import pytest

import microsim.schema as ms
from microsim.schema.optical_config.lib import FITC

MATSLINES = ms.MatsLines(density=0.4, length=30, azimuth=5, max_r=1)
GREEN_MATSLINES = ms.FluorophoreDistribution(distribution=MATSLINES)
NA1_4 = ms.ObjectiveLens(numerical_aperture=1.4)
CONFOCAL_AU0_2 = ms.Confocal(pinhole_au=0.2)
WIDEFIELD = ms.Widefield()


def test_simulation_json_schema() -> None:
    """Ensure the Simulation model can be cast to JSON schema."""
    assert isinstance(ms.Simulation.model_json_schema(), dict)


def test_model_dump(sim1: ms.Simulation) -> None:
    assert isinstance(sim1.model_dump(mode="python"), dict)
    assert isinstance(sim1.model_dump(mode="json"), dict)
    assert isinstance(sim1.model_dump_json(), str)


@pytest.mark.parametrize("precision", ["f4", "f8"])
@pytest.mark.parametrize("seed", [None, 100])
@pytest.mark.parametrize("modality", [WIDEFIELD, CONFOCAL_AU0_2], ids=lambda x: x.type)
def test_schema(
    sim1: ms.Simulation,
    np_backend: ms.BackendName,
    modality: ms.Modality,
    tmp_path: Path,
    precision: str,
    seed: int | None,
) -> None:
    sim1.settings.np_backend = np_backend
    sim1.settings.float_dtype = precision  # type: ignore
    sim1.modality = modality
    sim1.output_path = tmp_path / "output.zarr"
    sim1.settings.random_seed = seed

    out1 = sim1.run()
    assert sim1.output_space
    assert out1.squeeze().shape == sim1.output_space.shape
    assert sim1.ground_truth().dtype == np.dtype(precision)

    # ensure we have the right datatype
    if np_backend == "jax":
        # for jax, we will end up with the xarray_jax.JaxArrayWrapper
        assert "xarray_jax" in type(out1.data).__module__
    else:
        assert type(out1.data).__module__.split(".")[0].startswith(np_backend)

    out2 = sim1.run()
    if hasattr(out1.data, "get"):
        out1 = out1.copy(data=out1.data.get(), deep=False)
    if hasattr(out2.data, "get"):
        out2 = out2.copy(data=out2.data.get(), deep=False)
    if seed is None and np_backend != "jax":
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
        output_path=tmp_path / f"output.{ext}",
    )
    sim.run()


def test_sim_from_json() -> None:
    json_string = """
    {
        "truth_space": {
            "shape": [128, 512, 512],
            "scale": [0.02, 0.01, 0.01]
        },
        "output_space": {
            "downscale": 8
        },
        "sample": {
            "labels": [
            {
                "type": "matslines",
                "density": 0.5,
                "length": 30,
                "azimuth": 5,
                "max_r": 1
            }
            ]
        },
        "modality": {
            "type": "confocal",
            "pinhole_au": 0.2
        }
    }
    """

    ms.Simulation.model_validate_json(json_string)


def test_simulation_from_ground_truth() -> None:
    ground_truth = np.random.rand(64, 128, 128)
    scale = (0.04, 0.02, 0.02)
    sim = ms.Simulation.from_ground_truth(ground_truth=ground_truth, scale=scale)
    assert sim.truth_space.scale == scale
    sim_truth = sim.ground_truth().squeeze()
    if hasattr(sim_truth.data, "get"):
        sim_truth = sim_truth.data.get()
    np.testing.assert_array_almost_equal(sim_truth, ground_truth)


def test_pickle(sim1: ms.Simulation) -> None:
    pickled = pickle.dumps(sim1)
    assert pickle.loads(pickled) == sim1
    assert sim1.model_copy(deep=True) is not sim1
