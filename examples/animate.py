import numpy as np
import tifffile

from microsim import schema as ms
from microsim.schema.optical_config import lib

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(72, 1024, 1024), scale=(0.032, 0.032, 0.032)),
    output_space=ms.DownscaledSpace(downscale=4),
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="np_seg"),
                fluorophore="DAPI",
                concentration=3,
            ),
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_seg"),
                fluorophore="EGFP",
                concentration=2,
            ),
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_seg"),
                fluorophore="mCherry",
            ),
        ]
    ),
    channels=[lib.DAPI, lib.FITC, lib.DSRED],
    modality=ms.Confocal(pinhole_au=1.6),
    detector=ms.CameraCCD(qe=0.85, read_noise=6),
    settings=ms.Settings(max_psf_radius_aus=2),
    exposure_ms=3,
)

# colormaps = ("cyan", "green", "magenta")
# ground_truth = sim.ground_truth()
# animate(
#     ground_truth,
#     "gt_r.mp4",
#     colorized_axis=Axis.F,
#     colormaps=("gray_r", "gray_r", "gray_r"),
#     upsize=2,
#     fps=10,
# )
# animate(
#     ground_truth,
#     "gt.mp4",
#     colorized_axis=Axis.F,
#     colormaps=colormaps,
#     upsize=2,
#     fps=10,
# )
# optical_image = sim.optical_image()
# animate(optical_image, "oi.mp4", colormaps=colormaps, upsize=2, fps=10)
# digital_image = sim.digital_image()
# animate(digital_image, "di.mp4", colormaps=colormaps, upsize=4)

aus = []
for pinhole in (0.2, 0.6, 1.2, 2.0, 4.0, 8.0, 20):
    sim.exposure_ms = 6 * pinhole**0.4
    sim.modality.pinhole_au = pinhole
    aus.append(sim.digital_image())
combo = np.stack(aus)
tifffile.imwrite("pinhole_aus.ome.tif", combo, metadata={"axes": "TCZYX"}, ome=True)
