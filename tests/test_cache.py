from unittest.mock import patch

import xarray.testing as xrt

import microsim.schema as ms
from microsim import _data_array


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
