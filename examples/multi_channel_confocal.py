import time

from microsim import schema as ms
from microsim.schema.optical_config.lib import EYFP, FITC
from microsim.util import ndview

# short stubby lines labeled in EGFP and long thin lines labeled in EYFP
# bleedthrough of the EGFP signal in to the EYFP is more visible than vice versa

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(64, 256, 256), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=1, length=8, azimuth=1, max_r=1),
                fluorophore="EGFP",
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=0.1, length=30, azimuth=50, max_r=1),
                fluorophore="EYFP",
            ),
        ]
    ),
    modality=ms.Confocal(pinhole_au=1),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=4),
    detector=ms.CameraCCD(qe=0.82, read_noise=4, bit_depth=12),
    channels=[FITC, EYFP],
    # output_path="au1.tif",
)

_start = time.time()
result = sim.run()
print(time.time() - _start)

ndview(result)
