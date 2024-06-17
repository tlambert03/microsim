from collections import defaultdict

import numpy as np
import xarray as xr

from .interval_creation import Bin, generate_bins
from .schema.dimensions import Axis

_BIN_CACHE: defaultdict[str, defaultdict[int, dict[str, list[Bin]]]] = defaultdict(
    lambda: defaultdict(dict)
)


def bin_events(
    fluor: str,
    ex_filter: str,
    num_bins: int,
    em_wavelengths: np.ndarray,
    em_events: np.ndarray,
) -> xr.DataArray:
    """Bin the emission data into the given number of bins."""
    # cache = _BIN_CACHE[fluor][num_bins]
    # if (bins := cache.get(ex_filter)) is None:
    # cache[ex_filter] =
    bins = generate_bins(em_wavelengths, em_events, num_bins)
    bins = sorted(set([bins.start for bins in bins] + [bins[-1].end]))
    bins_arr = np.asarray(bins)
    data = xr.DataArray(em_events, dims=[Axis.W], coords={Axis.W: em_wavelengths})
    return data.groupby_bins(data[Axis.W], bins=bins_arr).sum()
