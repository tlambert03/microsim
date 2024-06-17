from microsim import schema as ms
from microsim.schema.optical_config import lib
from microsim.util import ndview

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(52, 512, 512), scale=(0.064, 0.064, 0.064)),
    output_space={"downscale": 2},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.Cosem(dataset="jrc_hela-3", image="er-mem_pred"),
                fluorophore="EGFP",
            ),
            ms.FluorophoreDistribution(
                distribution=ms.Cosem(dataset="jrc_hela-3", image="mito-mem_pred"),
                fluorophore="mCherry",
            ),
        ]
    ),
    channels=[lib.FITC, lib.DSRED],
    modality=ms.Widefield(),
    detector=ms.CameraCCD(qe=0.82, read_noise=6),
    output_path="h2-cf.tif",
    settings=ms.Settings(max_psf_radius_aus=2),
)

ndview(sim.run())
