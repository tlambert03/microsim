from rich import print

from microsim import schema as ms

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
                fluorophore="EGFP",
            )
        ]
    ),
    modality=ms.Confocal(),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=4),
    detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12),
)

illum = sim.illumination_flux()
print(illum.shape)
print(illum.sizes)
