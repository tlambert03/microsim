from pathlib import Path

import numpy as np

from microsim.illum._sim import SIMIllum3D
from microsim.util import uniformly_spaced_xarray


def test_new_model():
    REF = np.load(Path(__file__).parent / "illum_ref.npy")
    illum = SIMIllum3D()
    space = uniformly_spaced_xarray(shape=(24, 32, 48), scale=(0.01, 0.01, 0.01))
    result = illum.render(space)
    checksum = result.sum("P").sum("A").astype(np.float32)
    if hasattr(checksum, "get"):
        np.testing.assert_allclose(REF, checksum.get(), rtol=1e-6)
    else:
        np.testing.assert_allclose(REF, checksum, rtol=1e-6)
