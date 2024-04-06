import numpy as np

from microsim.schema.sample import MatsLines
from microsim.util import uniformly_spaced_xarray


def test_matslines():
    array = uniformly_spaced_xarray((256, 1024, 1024))
    d = MatsLines().render(array)
    assert np.max(d) > 0
