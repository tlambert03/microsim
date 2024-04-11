import time

from microsim import schema as ms
from microsim.util import ortho_plot

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.02, 0.01, 0.01)),
    output_space={"downscale": 8},
    sample=ms.Sample(labels=[ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1)]),
    modality=ms.Confocal(pinhole_au=0.5),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
    detector=ms.CameraCCD(qe=0.82, read_noise=6, bit_depth=12),
    # output_path="au1.tif",
)

_start = time.time()
result = sim.run()
print(time.time() - _start)
ortho_plot(result.data)
