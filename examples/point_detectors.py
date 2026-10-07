"""Compare point-scanning detectors (PMT vs HyD) on a confocal simulation.

For point detectors, `exposure_ms` is the pixel dwell time.
"""

import matplotlib.pyplot as plt

from microsim import schema as ms
from microsim.schema.detectors import lib
from microsim.schema.optical_config.lib import FITC

DWELL_MS = 0.002  # 2 µs pixel dwell

sim = ms.Simulation(
    truth_space={"upscale": 4},
    output_space=ms.ShapeScaleSpace(shape=(1, 128, 128), scale=(0.16, 0.08, 0.08)),
    sample=[
        ms.FluorophoreDistribution(
            distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
            fluorophore="EGFP",
            concentration=5,
        )
    ],
    # NOTE: real confocal irradiances approach ~1 MW/cm^2, but microsim does not yet
    # model excitation saturation, so emission scales linearly with power and would
    # be ~200x too bright.  5 kW/cm^2 is a stand-in that yields realistic detector
    # count rates (tens of Mcps at the brightest pixels).
    channels=[FITC.model_copy(update={"power": 5_000})],  # W/cm^2
    modality=ms.Confocal(pinhole_au=1),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
)
optical_image = sim.optical_image()

detectors = {
    "GaAsP PMT (gain=20)": lib.PMT_GAASP.model_copy(update={"gain": 20}),
    "GaAsP PMT (gain=100, saturated)": lib.PMT_GAASP.model_copy(update={"gain": 100}),
    "HyD counting": lib.HYD_SP8,
    "HyD counting, 4x averaging": lib.HYD_SP8.model_copy(update={"averaging": 4}),
}

fig, axes = plt.subplots(1, len(detectors), figsize=(4 * len(detectors), 4))
for ax, (label, detector) in zip(axes, detectors.items(), strict=True):
    sim.detector = detector
    img = sim.digital_image(optical_image, exposure_ms=DWELL_MS)
    ax.imshow(img[0, 0], cmap="gray")
    ax.set_title(f"{label}\nmax={int(img.max())}")
    ax.axis("off")
fig.tight_layout()
plt.show()
