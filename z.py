import matplotlib.pyplot as plt

import microsim.schema as ms
from microsim.interval_creation import bin_spectrum
from microsim.schema._emission import get_emission_events

channel = ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Green")
fluor = ms.Fluorophore.from_fpbase("EGFP")

r = channel.absorption_rate(fluor)
r2 = r.groupby_bins("w", bins=24).sum()
em_spectrum = get_emission_events(channel, fluor)
binned_events = bin_spectrum(
    spectrum=em_spectrum,
    bins=None,  # TODO: use the same bins as illumination?
    num_bins=3,  # TODO: same num_bins as illumination?
    binning_strategy="equal_space",  # to be consistent with PR#35
) / 1e9
r.plot()
r2.plot()
binned_events.plot()
plt.show()
