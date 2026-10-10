"""Compare point-scanning detectors (PMT vs HyD) on a confocal simulation.

For point detectors, `exposure_ms` is the pixel dwell time.
"""

import matplotlib.pyplot as plt

from microsim import schema as ms
from microsim.schema.detectors import lib
from microsim.schema.optical_config.lib import FITC

DWELL_MS = 0.001  # 1 µs pixel dwell

sim = ms.Simulation(
    truth_space={"upscale": 4},
    output_space=ms.ShapeScaleSpace(shape=(1, 128, 128), scale=(0.16, 0.08, 0.08)),
    sample=[
        ms.FluorophoreDistribution(
            distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
            fluorophore="EGFP",
            concentration=1,
        )
    ],
    # peak irradiance at focus (~25 µW in a diffraction-limited spot).  Gives tens
    # of Mcps at the brightest pixels; EGFP saturation is minor here (k*tau ~ 0.03)
    channels=[FITC.model_copy(update={"power": 2e4})],  # W/cm^2
    modality=ms.Confocal(pinhole_au=1),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
    # QE is applied when the optical image is computed (see microsim#152).  All the
    # detectors below are GaAsP (peak QE 0.45, nearly identical spectra), so one
    # optical image serves them all.
    detector=lib.HYD_SP8,
)
optical_image = sim.optical_image()

detectors = {
    # hv is the control voltage of a Hamamatsu module (0.5-0.9 V).  At 0.9 V the
    # brightest pixels exceed the module's 2 µA maximum output current.
    "GaAsP PMT (hv=0.6 V)": lib.PMT_GAASP.model_copy(update={"hv": 0.6}),
    "GaAsP PMT (hv=0.9 V, saturated)": lib.PMT_GAASP.model_copy(update={"hv": 0.9}),
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
