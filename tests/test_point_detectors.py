from __future__ import annotations

import numpy as np
import pytest

from microsim._data_array import DataArray
from microsim.schema import PMT, HyD, Simulation
from microsim.schema.backend import NumpyAPI
from microsim.schema.detectors import lib


def _flat(photons_per_s: float, shape: tuple[int, ...] = (1, 1, 200, 200)) -> DataArray:
    return DataArray(np.full(shape, photons_per_s), dims=("c", "z", "y", "x"))


@pytest.fixture
def xp() -> NumpyAPI:
    xp = NumpyAPI()
    xp.set_random_seed(0)
    return xp


def test_hyd_counting_low_rate(xp: NumpyAPI) -> None:
    # 1 Mcps x 10 µs = 10 photons; dead-time loss is ~0.15%
    det = HyD(dark_current=0, bit_depth=16)
    out = det.render(_flat(1e6), exposure_ms=0.01, xp=xp)
    assert out.dtype == np.uint16
    assert np.mean(out) == pytest.approx(10, rel=0.02)
    assert np.var(out) == pytest.approx(10, rel=0.05)  # Poisson


def test_hyd_dead_time_saturation(xp: NumpyAPI) -> None:
    # 200 Mcps with 1.5 ns non-paralyzable dead time -> 200 / (1 + 0.3) Mcps
    det = HyD(dark_current=0, dead_time_ns=1.5, bit_depth=16)
    out = det.render(_flat(2e8), exposure_ms=0.001, xp=xp)
    assert np.mean(out) == pytest.approx(200 / 1.3, rel=0.01)


def test_hyd_standard_mode_and_averaging(xp: NumpyAPI) -> None:
    det = HyD(mode="standard", gain=3, averaging=4, dead_time_ns=0)
    out = det.render(_flat(1e6), exposure_ms=0.01, xp=xp)
    assert np.mean(out) == pytest.approx(30, rel=0.02)
    # averaging 4 passes -> variance of mean is (gain^2 * 4*lam) / 16
    assert np.var(out) == pytest.approx(9 * 40 / 16, rel=0.1)


def test_pmt_excess_noise(xp: NumpyAPI) -> None:
    # mean = gain * lam;  var = enf * gain^2 * lam (+ read noise, rounding)
    det = PMT(gain=10, enf=1.5, read_noise=0, offset=0, bit_depth=16)
    out = det.render(_flat(1e6), exposure_ms=0.01, xp=xp)
    assert np.mean(out) == pytest.approx(100, rel=0.02)
    assert np.var(out) == pytest.approx(1.5 * 100 * 10, rel=0.1)


def test_pmt_offset_and_clipping(xp: NumpyAPI) -> None:
    det = PMT(gain=10, offset=-1000, bit_depth=8)
    assert np.all(det.render(_flat(1e6), exposure_ms=0.01, xp=xp) == 0)
    det = PMT(gain=1000, bit_depth=8)
    assert np.all(det.render(_flat(1e7), exposure_ms=0.01, xp=xp) == 255)


def test_point_detectors_in_simulation() -> None:
    for det in (lib.PMT_GAASP, lib.HYD_SP8):
        sim = Simulation.model_validate(
            {
                "truth_space": {"shape": (8, 32, 32), "scale": (0.04, 0.02, 0.02)},
                "output_space": {"downscale": 2},
                "sample": [],
                "detector": det.model_dump(),
            }
        )
        assert type(sim.detector) is type(det)
