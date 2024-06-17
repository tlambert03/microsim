from microsim import schema as ms
from microsim.schema.optical_config import lib
from microsim.util import ndview

sim = ms.Simulation(
    # note: this is a rather coarse simulation, but it's fast
    # scale should be a one of .004 * 2^n, where n is an integer from 0 to 4
    # space basically determines the field of view.
    truth_space=ms.ShapeScaleSpace(shape=(52, 512, 512), scale=(0.064, 0.064, 0.064)),
    output_space={"downscale": 2},
    sample=ms.Sample(
        labels=[
            # pick dataset and layer name from https://openorganelle.janelia.org/datasets
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
                fluorophore="EGFP",
            ),
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_pred"),
                fluorophore="mCherry",
            ),
        ]
    ),
    channels=[lib.FITC, lib.DSRED],
    modality=ms.Widefield(),
    detector=ms.CameraCCD(qe=0.82, read_noise=6),
    settings=ms.Settings(max_psf_radius_aus=2),
)

ndview(sim.digital_image())
