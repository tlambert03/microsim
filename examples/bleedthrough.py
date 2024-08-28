from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tifffile as tf

from microsim import schema as ms
from microsim.cosem import CosemDataset

EGFP = ms.Fluorophore.from_fpbase("EGFP")
VENUS = ms.Fluorophore.from_fpbase("Venus")
RATIO = 1.5


def run_simulation(dset: str) -> None:
    print("Running simulation for", dset)
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
                    fluorophore=EGFP,
                ),
                ms.FluorophoreDistribution(
                    distribution=ms.CosemLabel(dataset=dset, label="mito-mem_pred"),
                    fluorophore=VENUS,
                    concentration=RATIO,
                ),
            ],
            channels=["i6WL::Widefield Dual Green", "i6WL::Widefield Triple Yellow"],
            modality=ms.Confocal(pinhole_au=1.5),
            detector=ms.CameraCCD(qe=0.82, read_noise=2),
            settings=ms.Settings(max_psf_radius_aus=2, cache=False),
            exposure_ms=0.5,
            # output_path="bleedout.tif",
        )
    except Exception as e:
        print("❌ FAILED", dset, str(e))
        return

    dest = Path("egfp_er_venus_mito") / dset
    dest.mkdir(parents=True, exist_ok=True)

    oipf = sim.optical_image_per_fluor()
    with_bleed = sim.digital_image(oipf.sum("f"))
    tf.imwrite(
        dest / f"{dset}_bleedthrough_{RATIO}.tif",
        with_bleed.transpose("z", "c", "y", "x"),
        imagej=True,
    )
    just_egfp = sim.digital_image(oipf.sel(f=EGFP))
    tf.imwrite(dest / f"{dset}_just_egfp_{RATIO}.tif", just_egfp.isel(c=0), imagej=True)
    just_venus = sim.digital_image(oipf.sel(f=VENUS))
    tf.imwrite(
        dest / f"{dset}_just_venus_{RATIO}.tif", just_venus.isel(c=1), imagej=True
    )


if __name__ == "__main__":
    dsets = CosemDataset.names()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_simulation, dsets))
