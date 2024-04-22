import napari
import numpy as np

from microsim import schema as ms
from microsim.util import downsample

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(65, 1024, 1024), scale=(0.025, 0.01, 0.01)),
    output_space=ms.ShapeScaleSpace(shape=(13, 128, 128), scale=(0.125, 0.08, 0.08)),
    sample=ms.Sample(labels=[ms.MatsLines()]),
    modality=ms.SIM3D(angles=[0], nphases=1),
    detector=ms.CameraCCD(
        qe=0.82, full_well=18000, read_noise=6, bit_depth=12, offset=100
    ),
    settings=ms.Settings(random_seed=100),
    output_path="au1.tif",
)

ground_truth = sim.ground_truth()

channel = sim.channels[0]
dig = sim.modality.digital_image(
    ground_truth,
    channel=channel,
    objective_lens=sim.objective_lens,
    settings=sim.settings,
    outspace=sim.output_space,
)


img = np.asarray([downsample(x, 4) for x in dig])

napari.imshow(img)
napari.run()
