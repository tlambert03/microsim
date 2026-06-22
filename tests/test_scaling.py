import numpy as np

from microsim import schema as ms


def test_downscale_truth() -> None:
    sample = [
        ms.FluorophoreDistribution(
            distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
        )
    ]

    sim_down = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(64, 256, 256), scale=(0.04, 0.02, 0.02)),
        output_space={"downscale": 4},
        sample=sample,
        settings=ms.Settings(random_seed=100, max_psf_radius_aus=1),
    )
    sim_up = ms.Simulation(
        truth_space={"upscale": 4},
        output_space=ms.ShapeScaleSpace(shape=(16, 64, 64), scale=(0.16, 0.08, 0.08)),
        sample=sample,
        settings=ms.Settings(random_seed=100, max_psf_radius_aus=1),
    )

    # NB: compare via numpy — xarray's .equals/.identical can misbehave on
    # JAX-backed (xarray_jax-wrapped) data arrays.
    np.testing.assert_array_equal(
        np.asarray(sim_down.ground_truth().data),
        np.asarray(sim_up.ground_truth().data),
    )
    np.testing.assert_allclose(
        np.asarray(sim_down.digital_image().data),
        np.asarray(sim_up.digital_image().data),
    )
