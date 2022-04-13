from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import poisson

if TYPE_CHECKING:
    from ..models import Camera, CameraCMOS, CameraEMCCD


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


def bin(array: np.ndarray, binfactor: int, method="sum") -> np.ndarray:
    assert array.ndim == 2, "Cannot yet bin >2D images"
    m, n = array.shape
    f = getattr(np, method)
    reshaped = np.reshape(array, (m // binfactor, binfactor, n // binfactor, binfactor))
    return f(f(reshaped, 3), 1)
