from microsim import schema as ms
from microsim.schema.optical_config import lib

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(48, 512, 512), scale=(0.032, 0.032, 0.032)),
    output_space={"downscale": 2},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
                fluorophore="alexa fluor 647",
            ),
        ]
    ),
    channels=[lib.CY5],
    modality=ms.Confocal(pinhole_au=2.0),
    detector=ms.CameraCCD(qe=0.82, read_noise=2),
    settings=ms.Settings(max_psf_radius_aus=2),
)


result = sim.digital_image()
import ndv

ndv.imshow(result)

# save just the central slice to a png
from imageio.v3 import imwrite

central = result[result.shape[0] // 2]

imwrite("x_output.png", (central / central.max() * 65535).astype("uint16"))
