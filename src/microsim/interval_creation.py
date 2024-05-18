"""In this file, a spectrum is divided into intervals. AreaBasedInterval class ensures that the area under the curve is equal for all intervals."""

from bisect import bisect_left

import numpy as np

from microsim.schema._base_model import SimBaseModel


class Bin(SimBaseModel):
    """One interval."""

    start: float | None = None
    end: float | None = None
    mean: float | None = None
    mode: float | None = None


class BaseInterval(SimBaseModel):
    num_clusters: int
    bins: list[Bin] | None = None

    def generate_bins(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def get_bin_probablities(self, value) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class AreaBasedInterval(BaseInterval):
    """Every interval has equal area under the curve."""

    num_clusters: int

    def generate_bins(self, x: np.ndarray, y: np.ndarray) -> list[Bin]:
        self.bins = []

        cumsum = np.cumsum(y)
        step = cumsum[-1] / self.num_clusters
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
            self.bins.append(Bin(start=x[start_idx], end=x[end_idx], mean=x[mid_idx]))
            start_val = end_val
