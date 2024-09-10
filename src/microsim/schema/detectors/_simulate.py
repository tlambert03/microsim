from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from microsim.schema.backend import NumpyAPI
from microsim.schema.detectors import Camera, CameraCMOS, CameraEMCCD
from microsim.util import bin_window

if TYPE_CHECKING:
    from microsim._data_array import ArrayProtocol, xrDataArray


def simulate_camera(
    camera: Camera,
    image: xrDataArray,
    exposure_ms: float | xrDataArray = 100,
    binning: int = 1,
    add_poisson: bool = True,
    xp: NumpyAPI | None = None,
) -> ArrayProtocol:
    """Simulate camera detection.

    Parameters
    ----------
    camera : Camera
        camera objects
    image : DataArray
        array where each element represents photons / second
    exposure_ms : float | DataArray
        exposure time in milliseconds
    binning: int
        camera binning
    add_poisson: bool
        Whether to include poisson noise.
    xp: NumpyAPI
        Numpy API provider.

    Returns
    -------
    DataArray
        simulated image with camera and poisson noise
    """
    xp = NumpyAPI.create(xp)

    exposure_s = exposure_ms / 1000
    incident_photons = image * exposure_s
    # restrict to positive values
    incident_photons = xp.maximum(incident_photons.data, 0)

    # sample poisson noise
    if add_poisson:
        # FIXME: commenting this out since we also apply it in filtered_emission...
        # need to reconcile this
        # incident_photons = incident_photons * camera.qe
        detected_photons = xp.poisson_rvs(
            incident_photons, shape=incident_photons.shape
        )

    # dark current
    avg_dark_e = camera.dark_current * exposure_s + camera.clock_induced_charge
    if not isinstance(avg_dark_e, float):
        new_shape = avg_dark_e.shape + (1,) * (detected_photons.ndim - 1)
        avg_dark_e = np.asarray(avg_dark_e).reshape(new_shape)  # type: ignore [assignment]
    thermal_electrons = xp.poisson_rvs(avg_dark_e, shape=detected_photons.shape)
    total_electrons = detected_photons + thermal_electrons

    # cap total electrons to full-well-capacity
    total_electrons = xp.minimum(total_electrons, camera.full_well)

    if binning > 1 and not isinstance(camera, CameraCMOS):
        total_electrons = bin_window(total_electrons, binning, "sum")

    # add em gain
    if isinstance(camera, CameraEMCCD):
        total_electrons = camera.apply_em_gain(total_electrons)

    # model read noise
    gray_values = camera.quantize_electrons(total_electrons, xp)

    # sCMOS binning
    if binning > 1 and isinstance(camera, CameraCMOS):
        gray_values = bin_window(gray_values, binning, "mean")

    # ADC saturation
    gray_values = xp.minimum(gray_values, camera.max_intensity)
    if camera.bit_depth > 16:
        output = gray_values.astype("uint32")
    if camera.bit_depth > 8:
        output = gray_values.astype("uint16")
    else:
        output = gray_values.astype("uint8")

    return output
