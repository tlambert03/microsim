import numpy as np

from microsim.illum._spinning_disc import pinhole_coords


def test_pinhole_coords():
    points = pinhole_coords()
    e = np.array(
        [[13.971192, -5.459468], [14.068849, -5.226076], [14.162555, -4.991069]]
    )
    np.testing.assert_allclose(points[:3], e)
