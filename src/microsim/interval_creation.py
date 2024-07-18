"""Creating wavelength intervals/bins.

In this file, a spectrum is divided into intervals. AreaBasedInterval class
ensures that the area under the curve is equal for all intervals.
"""

from bisect import bisect_left
from typing import Literal, NamedTuple

import numpy as np
import xarray as xr

from microsim.schema.dimensions import Axis
from microsim.schema.spectrum import Spectrum


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


def generate_bins(
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    num_bins: int = 32,
    strategy: Literal["equal_area", "equal_space"] = "equal_space",
) -> list[Bin]:
    """Divide the spectrum into intervals."""
    if strategy == "equal_area":
        assert (
            y is not None
        ), "Can't generate equal area bins without intensities for the spectrum."
        return _generate_bins_equal_area(x, y, num_bins)
    elif strategy == "equal_space":
        return _generate_bins_equal_space(x, num_bins)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")


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


def _generate_bins_equal_space(x: np.ndarray, num_bins: int) -> list[Bin]:
    """Split the range of values in x into num_bins equally spaced bins.

    If len(x) is not divisible by num_bins, the len(x) % num_bins extra elements
    are distributed to the first len(x) % num_bins bins.
    """
    bins = []
    start = 0
    bin_size = len(x) // num_bins
    extra_elements = len(x) % num_bins
    for i in range(num_bins):
        extra_element = 1 if i < extra_elements else 0
        end = start + bin_size + extra_element
        bins.append(Bin(start=x[start], end=x[end - 1]))
        start = end

    return bins


def bin_spectrum(
    spectrum: Spectrum,
    bins: list[Bin] | None,
    *,
    num_bins: int = 64,
    binning_strategy: Literal["equal_area", "equal_space"] = "equal_space",
) -> xr.DataArray:
    """Bin the input spectrum into the given bins.

    If bins are not provided, generate them from the input spectrum
    and number of bins using the given binning strategy.

    Returns the binned spectrum as a `DataArray`.
    """
    wavelengths = spectrum.wavelength.magnitude
    if isinstance(spectrum.intensity, np.ndarray):
        intensities = spectrum.intensity
    else:
        intensities = spectrum.intensity.magnitude
    if bins is None:
        bins = generate_bins(
            x=wavelengths, y=intensities, num_bins=num_bins, strategy=binning_strategy
        )
    sbins = sorted(set([bins.start for bins in bins] + [bins[-1].end]))
    data = xr.DataArray(intensities, dims=[Axis.W], coords={Axis.W: wavelengths})
    binned_data = data.groupby_bins(data[Axis.W], bins=np.asarray(sbins)).sum()
    return binned_data.rename({f"{Axis.W}_bins": Axis.W})
