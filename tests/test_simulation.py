import importlib
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import microsim.schema as ms
from microsim.fft import convolve
from microsim.schema import simulation
from microsim.schema.dimensions import Axis
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


EXTENSIONS = [".zarr", ".nc"]
if importlib.util.find_spec("tifffile"):
    EXTENSIONS.append(".tiff")


@pytest.mark.parametrize("ext", EXTENSIONS)
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


def test_simulation_custom_distribution() -> None:
    """Test that we can use any class with a render method as a distribution."""

    class BadDistribution:
        def have_no_render_method(self, space, xp: ms.NumpyAPI | None = None):
            return space

    with pytest.raises(ValidationError):
        ms.Sample(labels=[BadDistribution()])

    class GoodDistribution:
        def render(self, space, xp: ms.NumpyAPI | None = None):
            return space

    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(64, 128, 128), scale=(0.2, 0.1, 0.1)),
        output_space={"downscale": 1},
        sample=ms.Sample(labels=[GoodDistribution()]),
    )
    sim.run()


def test_pickle(sim1: ms.Simulation) -> None:
    pickled = pickle.dumps(sim1)
    assert pickle.loads(pickled) == sim1
    assert sim1.model_copy(deep=True) is not sim1


def test_simulation_write_from_ground_truth(tmp_path: Path) -> None:
    """Test that unserializeable things don't prevent write."""
    sim = ms.Simulation.from_ground_truth(
        ground_truth=np.zeros((64, 128, 128)),
        scale=(0.04, 0.01, 0.01),
        output_space={"downscale": 4},
        output_path=tmp_path / "output.zarr",
    )
    sim.run()
    assert (tmp_path / "output.zarr").exists()


def test_reused_caches_across_simulations() -> None:
    """Simulations with the same optics reuse PSF transforms and emission rates."""
    convolve._KERNEL_FT_CACHE.clear()
    simulation._FLUORESCENCE_CACHE.clear()

    def _sim(seed: int) -> ms.Simulation:
        return ms.Simulation(
            truth_space=ms.ShapeScaleSpace(
                shape=(16, 32, 32), scale=(0.04, 0.02, 0.02)
            ),
            sample=ms.Sample(labels=[GREEN_MATSLINES]),
            objective_lens=NA1_4,
            channels=[FITC],
            settings=ms.Settings(random_seed=seed, cache=False),
        )

    sim_a, sim_b = _sim(1), _sim(2)
    img_a = sim_a.optical_image()
    assert len(convolve._KERNEL_FT_CACHE) == 1
    assert len(simulation._FLUORESCENCE_CACHE) == 1

    # the second simulation hits both caches...
    img_b = sim_b.optical_image()
    assert len(convolve._KERNEL_FT_CACHE) == 1
    assert len(simulation._FLUORESCENCE_CACHE) == 1
    # ... but with its own coordinate objects, and its own (different) sample
    rates_b = sim_b.filtered_emission_rates()
    assert rates_b.coords[Axis.C].values[0] is sim_b.channels[0]
    assert not np.allclose(img_a, img_b)
    # mutating a returned (cached) result doesn't corrupt the cache
    rates_b *= 0
    assert _sim(3).filtered_emission_rates().sum() > 0

    # and the result is the same as without the caches
    convolve._KERNEL_FT_CACHE.clear()
    simulation._FLUORESCENCE_CACHE.clear()
    np.testing.assert_allclose(_sim(2).optical_image(), img_b, rtol=1e-6)


def test_psf_cache_respects_float_dtype() -> None:
    """PSFs cached by a float64 simulation are not reused by a float32 one."""

    def _sim(dtype: str) -> ms.Simulation:
        return ms.Simulation(
            truth_space=ms.ShapeScaleSpace(shape=(8, 16, 16), scale=(0.04, 0.02, 0.02)),
            sample=ms.Sample(labels=[GREEN_MATSLINES]),
            settings=ms.Settings(float_dtype=dtype, cache=False),
        )

    assert _sim("float64").optical_image().dtype == np.float64
    assert _sim("float32").optical_image().dtype == np.float32


def test_run_batch(tmp_path: Path) -> None:
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(8, 16, 16), scale=(0.04, 0.02, 0.02)),
        output_space={"downscale": 2},
        sample=ms.Sample(labels=[GREEN_MATSLINES]),
        settings=ms.Settings(random_seed=1, cache=False),
        output_path=tmp_path / "out.zarr",
    )
    sim.run()  # populates (and must not leak) the template's ground truth
    samples = [ms.Sample(labels=[GREEN_MATSLINES]) for _ in range(3)]
    images = list(sim.run_batch(samples))

    assert len(images) == 3
    # same sample definition, but each gets its own seed -> different images
    assert not np.allclose(images[0], images[1])
    assert not np.allclose(images[0], sim.run())
    # reproducible
    for a, b in zip(images, sim.run_batch(samples), strict=True):
        np.testing.assert_array_equal(a, b)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.zarr",
        "out_0.zarr",
        "out_1.zarr",
        "out_2.zarr",
    ]


def test_changing_fields_resets_ground_truth() -> None:
    """model_copy(update=...) and field assignment don't reuse stale state."""

    def _sim() -> ms.Simulation:
        sim = ms.Simulation(
            truth_space=ms.ShapeScaleSpace(shape=(8, 16, 16), scale=(0.04, 0.02, 0.02)),
            output_space={"downscale": 2},
            sample=ms.Sample(labels=[GREEN_MATSLINES]),
            settings=ms.Settings(random_seed=1, cache=False),
        )
        sim.ground_truth()  # populate the cache
        return sim

    sim = _sim()
    truth = sim.ground_truth()
    other = {"labels": [{"distribution": {"type": "matslines", "density": 3}}]}

    copied = sim.model_copy(update={"sample": other})
    assert isinstance(copied.sample, ms.Sample)  # update is validated
    assert not truth.equals(copied.ground_truth())
    assert sim.model_copy().ground_truth().equals(truth)  # plain copy can reuse it

    sim.sample = other
    assert not truth.equals(sim.ground_truth())

    # relative spaces are re-resolved in the copy, without touching the original
    sim = _sim()
    big = ms.ShapeScaleSpace(shape=(8, 32, 32), scale=(0.04, 0.02, 0.02))
    copied = sim.model_copy(update={"truth_space": big})
    assert copied.output_space.shape == (4, 16, 16)
    assert sim.output_space.shape == (4, 8, 8)
    assert "truth_space" in copied.model_fields_set
    # ... also when assigning to a plain copy
    copied = sim.model_copy()
    copied.truth_space = big
    assert copied.output_space.shape == (4, 16, 16)
    assert sim.output_space.shape == (4, 8, 8)
    # ... and when swapping which space is relative
    out = ms.ShapeScaleSpace(shape=(4, 8, 8), scale=(0.08, 0.04, 0.04))
    copied = sim.model_copy(update={"truth_space": {"upscale": 2}, "output_space": out})
    assert copied.truth_space.shape == (8, 16, 16)


@pytest.mark.parametrize("modality", [ms.Widefield(), ms.Confocal(), ms.Identity()])
def test_modality_json_round_trip(modality: ms.Modality) -> None:
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(8, 32, 32), scale=(0.2, 0.1, 0.1)),
        sample=ms.Sample(labels=[MATSLINES]),
        modality=modality,
    )
    loaded = ms.Simulation.model_validate_json(sim.model_dump_json())
    assert type(loaded.modality) is type(modality)


def test_identity_modality_backend_conversion(np_backend: ms.BackendName) -> None:
    """Regression test for #124: em_rates must be converted to the active
    backend before use in Identity.render(), same as truth already is."""
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(8, 32, 32), scale=(0.2, 0.1, 0.1)),
        output_space={"downscale": 1},
        sample=ms.Sample(labels=[MATSLINES]),
        modality=ms.Identity(),
        settings=ms.Settings(np_backend=np_backend),
    )
    result = sim.optical_image()
    assert result.shape[1:] == sim.truth_space.shape
    flux = sim.emission_flux()
    assert flux.shape[2:] == sim.truth_space.shape
