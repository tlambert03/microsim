from microsim import schema as ms
from microsim.cosem import CosemDataset

dsets = CosemDataset.names()

for dset in dsets:
    try:
        sim = ms.Simulation(
            # note: this is a rather coarse simulation, but it's fast
            # scale should be a one of .004 * 2^n, where n is an integer from 0 to 4
            # space basically determines the field of view.
            truth_space=ms.ShapeScaleSpace(
                shape=(96, 1400, 1400), scale=(0.032, 0.032, 0.032)
            ),
            output_space={"downscale": 4},
            sample=[
                # pick dataset and layer name from https://openorganelle.janelia.org/datasets
                ms.FluorophoreDistribution(
                    distribution=ms.CosemLabel(dataset=dset, label="er-mem_pred"),
                    fluorophore="mEGFP",
                ),
                ms.FluorophoreDistribution(
                    distribution=ms.CosemLabel(dataset=dset, label="mito-mem_pred"),
                    fluorophore="Venus",
                    concentration=1.5,
                ),
            ],
            channels=["i6WL::Widefield Dual Green", "i6WL::Widefield Triple Yellow"],
            modality=ms.Confocal(pinhole_au=1.5),
            detector=ms.CameraCCD(qe=0.82, read_noise=2),
            settings=ms.Settings(max_psf_radius_aus=2, cache=False),
            exposure_ms=1,
            output_path="bleed/",
        )
    except Exception:
        continue

    sim.run()
