from typing import cast
import xarray as xr
from matplotlib import pyplot as plt

from microsim import schema as ms

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
                fluorophore="mNeonGreen",
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=0.1, length=1, azimuth=10, max_r=1),
                fluorophore="mScarlet",
            ),
        ]
    ),
    channels=[
        ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Green"),
        # ms.OpticalConfig(
        #     name="488nm",
        #     lights=[ms.LightSource.laser(wavelength=488)],
        # ),
        ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Red"),
    ],
    modality=ms.Confocal(),
    settings=ms.Settings(random_seed=100, max_psf_radius_aus=4),
    detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12),
)


def plot_summary(sim: ms.Simulation) -> None:
    nchannels = len(sim.channels)
    fig, ax = plt.subplots(2, nchannels, figsize=(12, 5))
    # optical configs
    for ch_idx in range(nchannels):
        spectra = [sim.channels[ch_idx].all_spectra()]
        # for lbl in sim.sample.labels:
        # spectra.append(lbl.fluorophore.all_spectra())
        xr.concat(spectra, dim="spectra").plot.line(ax=ax[0][ch_idx], x="w")

        # absorption rates
        rates = sim._absorption_rates().isel(c=ch_idx)
        rates.plot.line(ax=ax[1][ch_idx], x="w")

    for a in ax.flat:
        a.set_xlim(350, 750)
    plt.show()


plot_summary(sim)
