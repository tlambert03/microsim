from psfmodels._core import confocal_psf
from microsim.util import uniformly_spaced_xarray
from microsim.samples import MatsLines
from scipy import signal
import numpy as np
import tifffile
import uuid
from concurrent.futures import ProcessPoolExecutor


def _do_pinhole(args):
    pinhole, truth_shape, dz, dx, sample, sample_id = args

    print("working on pinhole: ", pinhole)

    psf = confocal_psf(
        pinhole_au=pinhole,
        z=truth_shape[0],
        nx=truth_shape[1],
        NA=1.4,
        ex_wvl=0.488,
        em_wvl=0.520,
        dz=dz,
        dxy=dx,
    )
    tifffile.imwrite(f"psf_{pinhole:0.2f}AU.tif", psf.astype("float32"))

    image = signal.convolve(sample, psf, mode="same")
    tifffile.imwrite(
        f"sample_{sample_id}_{pinhole:0.2f}AU.tif", image.astype("float32")
    )


if __name__ == "__main__":
    truth_shape = (48, 256, 256)
    dz = 0.08
    dx = 0.02

    truth_space = uniformly_spaced_xarray(shape=truth_shape, scale=(dz, dx, dx))
    sample = MatsLines(density=1, length=10, azimuth=10, max_r=0.9).render(truth_space)

    pinholes = np.linspace(0.1, 2.0, 20)

    sample_id = str(uuid.uuid4())[:8]
    tifffile.imwrite(f"sample_{sample_id}_truth.tif", sample.astype("float32"))

    with ProcessPoolExecutor(4) as executor:
        executor.map(
            _do_pinhole,
            [(p, truth_shape, dz, dx, sample, sample_id) for p in pinholes],
        )
