import pytest

from microsim.schema import Simulation


@pytest.fixture
def sim1():
    return Simulation(
        truth_space={"shape": (64, 128, 128), "scale": (0.02, 0.01, 0.01)},
        output_space={"downscale": 8},
        samples=[
            {
                "type": "matslines",
                "density": 0.4,
                "length": 30,
                "azimuth": 5,
                "max_r": 1,
            }
        ],
        objective_lens={"numerical_aperture": 1.4},
        channels=[
            {
                "name": "FITC",
                "excitation": 488,
                "emission": {"wavelength": 520, "bandwidth": 50},
            }
        ],
        modality={"type": "confocal", "pinhole": 0.2},
    )


@pytest.mark.parametrize("np_backend", ["numpy", "cupy", "jax"])
def test_schema(sim1: Simulation, np_backend: str, benchmark) -> None:
    sim1.settings.np_backend = np_backend

    output = benchmark(sim1.run)
    assert output.shape == sim1.output_space.shape
