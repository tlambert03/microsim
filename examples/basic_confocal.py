from microsim import schema as ms
from microsim.schema.detectors.lib import HYD_SP8
from microsim.schema.optical_config.lib import LASER_488
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
    # 488 nm laser, 1e5 W/cm^2 peak irradiance at the focus (~50 µW at NA 1.4)
    channels=[LASER_488],
    modality=ms.Confocal(pinhole_au=0.5),
    detector=HYD_SP8,
    exposure_ms=0.002,  # 2 µs pixel dwell time
    settings=ms.Settings(random_seed=100),
)

result = sim.run()
ortho_plot(result)
