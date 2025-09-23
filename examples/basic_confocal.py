from microsim import schema as ms
from microsim.util import ortho_plot

sim = ms.Simulation(
    truth_space={"upscale": 4},
    output_space=ms.ShapeScaleSpace(shape=(16, 64, 64), scale=(0.16, 0.08, 0.08)),
    sample=[
        ms.FluorophoreDistribution(
            distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
            fluorophore="EGFP",
            concentration=5,
        )
    ],
    exposure_ms=200,
    modality=ms.Confocal(pinhole_au=0.5),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
    detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12),
)

result = sim.run()
ortho_plot(result)
