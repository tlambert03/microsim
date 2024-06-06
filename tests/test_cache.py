from unittest.mock import patch

import numpy as np
import pytest
import xarray.testing as xrt

import microsim.schema as ms
from microsim import _data_array
from microsim.psf import cached_psf


def test_cache(sim1: ms.Simulation) -> None:
    with patch("microsim.schema.simulation.to_cache") as mock_to_cache:
        mock_to_cache.side_effect = _data_array.to_cache
        t1 = sim1.ground_truth()
        mock_to_cache.assert_called_once()

    # delete the cached attribute, to test the from_cache method
    del sim1._ground_truth

    with patch("microsim.schema.simulation.from_cache") as mock_to_cache:
        mock_to_cache.side_effect = _data_array.from_cache
        t2 = sim1.ground_truth()
        mock_to_cache.assert_called_once()

    xrt.assert_equal(t1, t2)


def test_cache_psf(caplog: pytest.LogCaptureFixture) -> None:
    kwargs = {
        "nz": 16,
        "nx": 64,
        "dx": 0.01,
        "dz": 0.02,
        "ex_wvl_um": 0.5,
        "em_wvl_um": 0.6,
        "objective": ms.ObjectiveLens(numerical_aperture=1.4),
        "pinhole_au": 1.0,
        "max_au_relative": None,
        "xp": ms.NumpyAPI(),
    }
    caplog.set_level(10)
    psf1 = cached_psf(**kwargs)
    cached_psf.cache_clear()  # clear the functools cache
    assert "Using cached PSF:" not in caplog.text
    psf2 = cached_psf(**kwargs)
    assert "Using cached PSF:" in caplog.text

    np.testing.assert_array_equal(psf1, psf2)
