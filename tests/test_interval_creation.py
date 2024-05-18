import pytest
from microsim.interval_creation import AreaBasedInterval
import numpy as np


def test_area_based_interval():
    x = np.arange(1,64*4)
    y = [2]*64 + [4]*32 + [8]*16 + [16]*8
    interval = AreaBasedInterval(num_clusters=4)
    interval.generate_bins(x, y)
    assert interval.bins[0].start == 1
    assert interval.bins[0].end == 64 -1

    assert interval.bins[1].start == 64
    assert interval.bins[1].end == 96 -1

    assert interval.bins[2].start == 96
    assert interval.bins[2].end == 112 -1

    assert interval.bins[3].start == 112
    assert interval.bins[3].end == 120