import time

from microsim import schema as ms
from microsim.schema.optical_config.lib import DAPI, FITC
from microsim.util import ortho_plot

GreenMats = ms.FluorophoreDistribution(
    distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
    fluorophore="EGFP",
)

Mcherry = ms.FluorophoreDistribution(
    distribution=ms.MatsLines(density=0.5, length=30, azimuth=50, max_r=1),
    fluorophore="mCherry",
)

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.02, 0.01, 0.01)),
    output_space={"downscale": 8},
    sample=ms.Sample(labels=[GreenMats, Mcherry]),
    modality=ms.Confocal(pinhole_au=0.5),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
    detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12),
    channels=[FITC, DAPI],
    # output_path="au1.tif",
)

_start = time.time()
result0 = sim.run(channel_idx=0)
result1 = sim.run(channel_idx=1)
print(time.time() - _start)
ortho_plot(result0.data)
ortho_plot(result1.data)
