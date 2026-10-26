import time

from microsim import schema as ms
from microsim.schema.detectors.lib import HYD_SP8
from microsim.schema.optical_config.lib import LASER_488, LASER_515
from microsim.util import ndview

# short stubby lines labeled in EGFP and long thin lines labeled in EYFP, imaged
# with 488 and 515 nm lasers at the same time (one track), so each fluorophore
# bleeds into the other's channel (in the 488 channel, EYFP even outshines EGFP).
# Try `tracks="sequential"`: 515 nm barely excites EGFP, so its bleedthrough into
# the EYFP channel mostly vanishes (EYFP still bleeds into the 488 channel, since
# 488 nm excites it too).

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
    settings=ms.Settings(random_seed=100),
    detector=HYD_SP8,
    exposure_ms=0.002,  # 2 µs pixel dwell time
    tracks="simultaneous",  # both lasers on, both detectors recording
    channels=[LASER_488, LASER_515],
    # output_path="au1.tif",
)

_start = time.time()
result = sim.run()
print(time.time() - _start)

ndview(result)
