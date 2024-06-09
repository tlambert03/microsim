import xarray as xr

from .interval_creation import generate_bins


class EmissionBins:
    """This class is used to generate and store the emission bins for a given fluorophore and emission filter.

    This can be cached for a given fluorophore, excitation filter and num bins.
    """

    data_dict = {}

    @staticmethod
    def get_bins(fluor: str, ex_filter: str, num_bins: int, em_wavelengths, em_events):
        if fluor in EmissionBins.data_dict:
            if ex_filter in EmissionBins.data_dict[fluor]:
                return EmissionBins.data_dict[fluor][num_bins][ex_filter]

        # generate it and store it.
        EmissionBins.data_dict[fluor] = EmissionBins.data_dict.get(fluor, {})
        EmissionBins.data_dict[fluor][num_bins] = EmissionBins.data_dict[fluor].get(
            num_bins, {}
        )
        EmissionBins.data_dict[fluor][num_bins][ex_filter] = generate_bins(
            em_wavelengths, em_events, num_bins
        )
        return EmissionBins.data_dict[fluor][num_bins][ex_filter]

    @staticmethod
    def bin_events(
        fluor: str, ex_filter: str, num_bins: int, em_wavelengths, em_events
    ) -> xr.DataArray:
        """Bin the emission data into the given number of bins."""
        bins = EmissionBins.get_bins(
            fluor, ex_filter, num_bins, em_wavelengths, em_events
        )
        bins_arr = [bins.start.magnitude for bins in bins]
        bins_arr.append(bins[-1].end.magnitude)
        data = xr.DataArray(em_events, dims=["w"], coords={"w": em_wavelengths})
        binned_events = data.groupby_bins(data["w"], bins=bins_arr).sum()
        return binned_events
