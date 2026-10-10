from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from microsim._data_array import DataArray
from microsim.schema import CameraCCD, CameraCMOS, CameraEMCCD
from microsim.schema.backend import NumpyAPI
from microsim.schema.detectors._detector import apply_multiplication_gain


def _flat(photons: float, shape: tuple[int, ...] = (1, 1, 200, 200)) -> DataArray:
    """Uniform image with `photons` per pixel in a 1 s exposure."""
    return DataArray(np.full(shape, photons), dims=("c", "z", "y", "x")[-len(shape) :])


@pytest.fixture
def xp() -> NumpyAPI:
    xp = NumpyAPI()
    xp.set_random_seed(0)
    return xp


def _ideal(cam_type: type, **kwargs: float) -> CameraCCD:
    """Camera with no read noise, dark signal, offset or clipping."""
    return cam_type(  # type: ignore[no-any-return]
        read_noise=0, dark_current=0, offset=0, bit_depth=32, full_well=10**9,
        electrons_per_adu=1, **kwargs,
    )  # fmt: skip


def test_ccd_is_poisson(xp: NumpyAPI) -> None:
    out = _ideal(CameraCCD).render(_flat(100), exposure_ms=1000, xp=xp)
    assert out.dtype == np.uint32
    assert np.mean(out) == pytest.approx(100, rel=0.01)
    assert np.var(out) == pytest.approx(100, rel=0.05)


def test_emccd_excess_noise(xp: NumpyAPI) -> None:
    # mean = G * n;  var = F * G^2 * n
    cam = _ideal(CameraEMCCD, em_gain=300)
    out = cam.render(_flat(20), exposure_ms=1000, xp=xp)
    assert np.mean(out) == pytest.approx(300 * 20, rel=0.01)
    var = cam.excess_noise_factor * 300**2 * 20
    assert np.var(out) == pytest.approx(var, rel=0.05)


def test_emccd_excess_noise_factor() -> None:
    def enf(g: float) -> float:
        return CameraEMCCD(em_gain=g).excess_noise_factor

    assert enf(1) == 1
    assert enf(2) == pytest.approx(1.5, abs=0.01)
    assert 1.97 < enf(300) < 2


def test_analog_gain() -> None:
    cam = CameraCCD(full_well=4095, bit_depth=12)
    assert cam.conversion_factor == 1  # default relative_gain = 1
    assert (
        CameraCCD(full_well=4095, bit_depth=12, relative_gain=2).conversion_factor
        == 0.5
    )
    assert CameraCCD(electrons_per_adu=0.25).conversion_factor == 0.25
    with pytest.raises(ValidationError, match="not both"):
        CameraCCD(relative_gain=1, electrons_per_adu=1)


def test_gain_deprecated() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CameraCCD()  # default: no warning
    with pytest.warns(FutureWarning, match="'gain' has been renamed"):
        cam = CameraCCD(gain=2)
    assert cam.relative_gain == 2
    with pytest.warns(FutureWarning, match="renamed to `relative_gain`"):
        assert cam.gain == 2
        assert CameraCCD().gain == 1  # default
        cam = CameraCCD(full_well=4095, bit_depth=12, electrons_per_adu=0.5)
        assert cam.gain == 2
    with pytest.warns(FutureWarning, match="renamed to `relative_gain`"):
        cam.gain = 3
    assert cam.relative_gain == 3
    assert cam.electrons_per_adu is None


def test_em_gain_at_least_one() -> None:
    with pytest.raises(ValidationError):
        CameraEMCCD(em_gain=0.5)


def test_multiplication_gain_fractional_input() -> None:
    # without shot noise, fractional electrons are multiplied without bias
    out = apply_multiplication_gain(np.full(100_000, 0.3), gain=100, enf=2)
    assert out.mean() == pytest.approx(30, rel=0.03)
    assert np.all(apply_multiplication_gain(np.zeros(10), gain=100, enf=2) == 0)


@pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 1, 64, 64), (64, 64)])
@pytest.mark.parametrize("cam_type", [CameraCCD, CameraEMCCD, CameraCMOS])
def test_binning(cam_type: type, shape: tuple[int, ...], xp: NumpyAPI) -> None:
    kwargs = {"em_gain": 10} if cam_type is CameraEMCCD else {}
    cam = _ideal(cam_type, **kwargs)
    out = cam.render(_flat(100, shape), exposure_ms=1000, binning=4, xp=xp)
    assert out.shape == (*shape[:-2], 16, 16)
    # charge binning (CCD) sums 16 pixels; digital binning (CMOS) averages them
    expected = 100 if cam_type is CameraCMOS else 1600 * cam.multiplication_gain
    assert np.mean(out) == pytest.approx(expected, rel=0.02)
