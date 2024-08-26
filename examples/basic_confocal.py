import time

from microsim import schema as ms
from microsim.util import ndview, ortho_plot

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(64, 256, 256), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
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
    # output_path="au1.tif",
)

_start = time.time()
result = sim.run()
print("ran in", time.time() - _start)
ortho_plot(result.data, show=True)
ndview(result)
