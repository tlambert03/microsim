"""Creating wavelength intervals/bins.

In this file, a spectrum is divided into intervals. AreaBasedInterval class
ensures that the area under the curve is equal for all intervals.
"""

from bisect import bisect_left
from typing import NamedTuple

import numpy as np


class Bin(NamedTuple):
    # TODO : include units for each of these. Use pint.
    """One interval."""

    start: float
    end: float
    mean: float | None = None
    mode: float | None = None

    def __contains__(self, x: object) -> bool:
        try:
            return self.start <= x <= self.end  # type: ignore
        except TypeError:
            return False

    def __str__(self) -> str:
        if self.start is not None:
            assert self.end is not None
            return f"[{self.start:.2f}-{self.end:.2f}]"
        elif self.mean is not None:
            return f"Mean:{self.mean:.2f}"
        else:
            assert self.mode is not None
            return f"Mode:{self.mode:.2f}"


def generate_bins(x: np.ndarray, y: np.ndarray, num_bins: int) -> list[Bin]:
    """Divide the spectrum into intervals."""
    return _generate_bins_equal_area(x, y, num_bins)


def _generate_bins_equal_area(x: np.ndarray, y: np.ndarray, num_bins: int) -> list[Bin]:
    bins = []
    cumsum = y.cumsum()
    step = cumsum[-1] / num_bins
    start_val = 0
    end_vals = np.arange(step, cumsum[-1], step)

    # Add the last bin if the last value is quite far from the last bin
    if cumsum[-1] - end_vals[-1] > step / 2:
        end_vals = np.append(end_vals, cumsum[-1])
    else:
        end_vals[-1] = cumsum[-1]

    for idx, end_val in enumerate(end_vals):
        mid_val = (start_val + end_val) / 2
        # NOTE: minus 1 because we want the disjoint intervals. Also, for the last
        # interval, we want the last index and so there is no minus 1.
        end_idx = bisect_left(cumsum, end_val) - 1 * (idx != len(end_vals) - 1)
        start_idx = bisect_left(cumsum, start_val)
        # TODO: mid is not the mean.
        mid_idx = bisect_left(cumsum, mid_val)
        bins.append(Bin(start=x[start_idx], end=x[end_idx], mean=x[mid_idx]))
        start_val = end_val
    return bins
