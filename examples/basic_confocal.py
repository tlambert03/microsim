import tifffile

from microsim import schema as ms
from microsim.util import animate

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(64, 512, 512), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
    sample=[
        ms.FluorophoreDistribution(
            distribution=ms.MatsLines(density=1.5, length=30, azimuth=5, max_r=1),
            fluorophore="EGFP",
            concentration=5,
        )
    ],
    exposure_ms=100,
    modality=ms.Confocal(pinhole_au=1.5),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
    detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12),
    # output_path="au1.tif",
)

result = sim.digital_image()
gt = sim.ground_truth()
tifffile.imwrite("gt.tif", gt.data)
di = sim.digital_image()

tifffile.imwrite("di.tif", di.data)
animate(gt, "gt.mp4", clims=(0, 2), fps=10)
animate(di, "di.mp4", upsize=4, fps=2.5)
