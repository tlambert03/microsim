from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.stats import poisson

from microsim.models import CameraCMOS, CameraEMCCD

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from microsim.models import Camera


def simulate_camera(
    camera: Camera,
    image: np.ndarray,
    exposure: float = 0.1,
    binning: int = 1,
    add_poisson: bool = True,
):
    """Simulate camera detection.

    Parameters
    ----------
    camera : Camera
        camera objects
    image : np.ndarray
        array where each element represents photons / second
    exposure : float
        exposure time in seconds
    binning: int
        camera binning
    add_poisson: bool
        Whether to include poisson noise.

    Returns
    -------
    np.ndarray
        simulated image with camera and poisson noise
    """
    incident_photons = image * exposure

    # sample poisson noise
    if add_poisson:
        detected_photons: np.ndarray = poisson.rvs(incident_photons * camera.qe)

    # dark current
    thermal_electrons: np.ndarray = poisson.rvs(
        camera.dark_current * exposure + camera.clock_induced_charge,
        size=detected_photons.shape,
    )
    total_electrons = detected_photons + thermal_electrons

    # cap total electrons to full-well-capacity
    total_electrons = np.minimum(total_electrons, camera.full_well)

    if binning > 1 and not isinstance(camera, CameraCMOS):
        total_electrons = bin(total_electrons, binning, "sum")

    # add em gain
    if isinstance(camera, CameraEMCCD):
        total_electrons = camera.apply_em_gain(total_electrons)

    # model read noise
    gray_values = camera.quantize_electrons(total_electrons)

    # sCMOS binning
    if binning > 1 and isinstance(camera, CameraCMOS):
        gray_values = bin(gray_values, binning, "mean")

    # ADC saturation
    gray_values = np.minimum(gray_values, camera.max_intensity)
    if camera.bit_depth > 16:
        return gray_values.astype("uint32")
    if camera.bit_depth > 8:
        return gray_values.astype("uint16")
    else:
        return gray_values.astype("uint8")


def bin(  # noqa: A001
    array: NDArray, factor: int | Sequence[int], method="sum", dtype=None
) -> NDArray:
    # TODO: deal with xarray
    f = getattr(np, method)
    binfactor = (factor,) * array.ndim if isinstance(factor, int) else factor
    new_shape = []
    for s, b in zip(array.shape, binfactor):
        new_shape.extend([s // b, b])
    reshaped = np.reshape(array, new_shape)
    for d in range(array.ndim):
        reshaped = f(reshaped, axis=-1 * (d + 1), dtype=dtype)
    return reshaped
