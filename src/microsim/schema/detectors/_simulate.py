from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from microsim.schema.backend import NumpyAPI
from microsim.schema.detectors import Camera, CameraCMOS, CameraEMCCD

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from microsim._data_array import ArrayProtocol


def simulate_camera(
    camera: Camera,
    image: ArrayProtocol,
    exposure_ms: float = 100,
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
    exposure_ms : float
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
    incident_photons = xp.maximum(incident_photons, 0)

    # sample poisson noise
    if add_poisson:
        detected_photons = xp.poisson_rvs(incident_photons * camera.qe)

    # dark current
    thermal_electrons = xp.poisson_rvs(
        camera.dark_current * exposure_s + camera.clock_induced_charge,
        shape=detected_photons.shape,
    )
    total_electrons = detected_photons + thermal_electrons

    # cap total electrons to full-well-capacity
    total_electrons = xp.minimum(total_electrons, camera.full_well)

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
    gray_values = xp.minimum(gray_values, camera.max_intensity)
    if camera.bit_depth > 16:
        output = gray_values.astype("uint32")
    if camera.bit_depth > 8:
        output = gray_values.astype("uint16")
    else:
        output = gray_values.astype("uint8")

    return output


def bin(  # noqa: A001
    array: npt.NDArray,
    factor: int | Sequence[int],
    method: str = "sum",
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray:
    # TODO: deal with xarray
    f = getattr(np, method)
    binfactor = (factor,) * array.ndim if isinstance(factor, int) else factor
    new_shape = []
    for s, b in zip(array.shape, binfactor, strict=False):
        new_shape.extend([s // b, b])
    reshaped = np.reshape(array, new_shape)
    for d in range(array.ndim):
        reshaped = f(reshaped, axis=-1 * (d + 1), dtype=dtype)
    return reshaped
