from .interval_creation import generate_bins


class EmissionBins:
    """This class is used to generate and store the emission bins for a given fluorophore and emission filter.

    This can be cached for a given fluorophore, excitation filter and num bins.
    """

    # TODO: One could ideally also have an emission filter here.

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
    ):
        """Bin the emission data into the given number of bins."""
        bins = EmissionBins.get_bins(
            fluor, ex_filter, num_bins, em_wavelengths, em_events
        )
        data = [0.0] * len(bins)
        cur_bin_idx = 0
        for wave_idx, wavelength in enumerate(em_wavelengths):
            while wavelength not in bins[cur_bin_idx]:
                cur_bin_idx += 1
                if cur_bin_idx >= len(bins):
                    break
            if cur_bin_idx >= len(bins):
                raise ValueError(f"Wavelength:{wavelength} not in any bin between \
                                 {bins[0]!s} and {bins[-1]!s}")

            data[cur_bin_idx] += em_events[wave_idx]
        return data, bins
