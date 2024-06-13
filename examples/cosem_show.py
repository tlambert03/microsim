from microsim import schema as ms
from microsim.util import ndview

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(
        shape=(128, 1280, 1280), scale=(0.032, 0.032, 0.032)
    ),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            # ms.Cosem(dataset="jrc_hela-3", image="chrom_seg"),
            ms.Cosem(dataset="jrc_hela-3", image="er-mem_pred"),
            # ms.Cosem(dataset="jrc_hela-3", image="er-mem_seg"),
            ms.Cosem(dataset="jrc_hela-3", image="mito-mem_pred"),
        ]
    ),
    modality=ms.Confocal(),
    detector=ms.CameraCCD(qe=0.82, read_noise=6),
    output_path="h2-cf.tif",
)

ndview(sim.run())
