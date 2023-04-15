import numpy as np
import pytest
from microsim.util import uniformly_spaced_xarray


def test_matslines():
    pytest.importorskip("microsim.samples.utils._bresenham")
    from microsim.samples import MatsLines

    array = uniformly_spaced_xarray((256, 1024, 1024))
    d = MatsLines().render(array)
    assert np.max(d) > 0
