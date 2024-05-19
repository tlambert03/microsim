"""In this file, a spectrum is divided into intervals. AreaBasedInterval class ensures that the area under the curve is equal for all intervals."""

from bisect import bisect_left
from typing import NamedTuple

import numpy as np

class Bin(NamedTuple):
    """One interval."""

    start: float | None = None
    end: float | None = None
    mean: float | None = None
    mode: float | None = None

def generate_bins(x: np.ndarray, y: np.ndarray, numbins:int) -> list[Bin]:
    """Divide the spectrum into intervals."""
    return _generate_bins_equal_area(x, y, numbins)

def _generate_bins_equal_area(x: np.ndarray, y: np.ndarray, numbins:int) -> list[Bin]:
    bins = []

    cumsum = np.cumsum(y)
    step = cumsum[-1] / numbins
    start_val = 0
    end_vals = np.arange(step, cumsum[-1], step)
    # Add the last bin if the last value is quite far from the last bin
    end_vals = np.append(end_vals, cumsum[-1])

    for idx, end_val in enumerate(end_vals):
        mid_val = (start_val + end_val) / 2
        # NOTE: minus 1 because we want the disjoint intervals. Also, for the last interval, we want the last index and so there is no minus 1.
        end_idx = bisect_left(cumsum, end_val) - 1 * (idx != len(end_vals) - 1)
        start_idx = bisect_left(cumsum, start_val)
        # TODO: mid is not the mean.
        mid_idx = bisect_left(cumsum, mid_val)
        bins.append(Bin(start=x[start_idx], end=x[end_idx], mean=x[mid_idx]))
        start_val = end_val
    
    return bins
