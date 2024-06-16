import numpy as np

from microsim.interval_creation import generate_bins


def test_area_based_interval():
    x = np.arange(1, 64 + 32 + 16 + 8 + 1)
    y = [2] * 64 + [4] * 32 + [8] * 16 + [16] * 8
    y = np.array(y)
    bins = generate_bins(x, y, 4)
    assert bins[0].start == 1
    assert bins[0].end == 64 - 1

    assert bins[1].start == 64
    assert bins[1].end == 96 - 1

    assert bins[2].start == 96
    assert bins[2].end == 112 - 1

    assert bins[3].start == 112
    assert bins[3].end == 120
