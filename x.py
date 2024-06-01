from microsim import schema as ms

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(
        shape=(480, 1024, 1024), scale=(0.016, 0.016, 0.016)
    ),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            ms.Cosem(dataset="jrc_hela-3", image="chrom_seg"),
            ms.Cosem(dataset="jrc_hela-3", image="mito-mem_seg"),
            ms.Cosem(dataset="jrc_hela-3", image="mt-out_seg"),
            ms.Cosem(dataset="jrc_hela-3", image="er_seg"),
        ]
    ),
    modality=ms.Confocal(),
    detector=ms.CameraCCD(
        qe=0.82, full_well=18000, read_noise=6, bit_depth=12, offset=100
    ),
    output_path="img.tif",
)

sim.run()
