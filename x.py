from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from matplotlib import pyplot as plt

from microsim import schema as ms
from microsim.schema.detectors._camera import ICX285
from microsim.schema.optical_config import Placement
from microsim.schema.optical_config.filter import bandpass


def spectral_detector(
    bins: int,
    min_wave: float,
    max_wave: float,
    lasers: Sequence[int],
    bp_bandwidth: float = 10,
) -> list[ms.OpticalConfig]:
    lights = [ms.LightSource.laser(laser) for laser in lasers]
    waves = np.arange(min_wave - 100, max_wave + 100, 1)

    # create fake bandpass ... could also be something like 80/20
    laser0, *rest = lasers
    bp = bandpass(waves, center=laser0, bandwidth=bp_bandwidth, transmission=1)
    for laser in rest:
        bp = bp + bandpass(waves, center=laser, bandwidth=bp_bandwidth)
    bp = ms.SpectrumFilter(
        transmission=ms.Spectrum(wavelength=waves, intensity=1 - bp),
        placement=Placement.BS,
    )

    configs: list[ms.OpticalConfig] = []
    edges = np.linspace(min_wave, max_wave, bins + 1)
    for i, (low, high) in enumerate(pairwise(edges)):
        mask = (waves >= low) & (waves <= high)
        f = ms.SpectrumFilter(
            transmission=ms.Spectrum(wavelength=waves, intensity=mask),
            placement=Placement.EM_PATH,
        )
        oc = ms.OpticalConfig(
            name=f"Channel {i}",
            lights=lights,
            filters=[bp, f],
        )
        configs.append(oc)
    return configs


sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(), fluorophore="mEGFP"
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(), fluorophore="mVenus"
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(), fluorophore="mCherry"
            ),
        ]
    ),
    channels=[
        ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Green"),
        ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Red"),
    ],
    # channels=spectral_detector(8, 480, 600, [488, 515, 561]),
    modality=ms.Confocal(),
    detector=ICX285,
)


def plot_summary(
    sim: ms.Simulation, transpose: bool = False, legend: bool = True
) -> None:
    nrows = 5
    ncols = len(sim.channels)
    if transpose:
        nrows, ncols = ncols, nrows
    fig, ax = plt.subplots(nrows, ncols, figsize=(18, 10), sharex=True)
    if transpose:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = ax.T
    else:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = ax
    if len(sim.channels) == 1:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = [fp_ax], [ex_ax], [ab_ax], [em_ax], [f_ax]

    for ch_idx, oc in enumerate(sim.channels):
        # FLUOROPHORES --------------------------------------
        for lbl in sim.sample.labels:
            if fluor := lbl.fluorophore:
                ex = fluor.absorption_cross_section
                ex.plot.line(ax=fp_ax[ch_idx], label=f"{fluor.name}")

        # ILLUMINATION PATH --------------------------------------
        ex_ax2 = ex_ax[ch_idx].twinx()
        for f in oc.filters:
            if f.placement == Placement.EM_PATH:
                continue

            spect = f.spectrum
            if f.placement == Placement.BS:
                spect = spect.inverted()
            ex_ax2.plot(spect.wavelength, spect.intensity, label=f"{f.name}", alpha=0.4)
        # light sources
        for light in oc.lights:
            ls = light.spectrum
            ex_ax2.plot(ls.wavelength, ls.intensity, label=f"{light.name}", alpha=0.4)

        # combined illumination
        full = oc.illumination_flux_density
        full.plot.line(ax=ex_ax[ch_idx], label="flux density", color="k")

        # ABSORPTION/EMISSION RATES --------------------------------------
        for lbl in sim.sample.labels:
            if fluor := lbl.fluorophore:
                rate = oc.absorption_rate(fluor)
                tot = rate.sum()
                rate.plot.line(
                    ax=ab_ax[ch_idx], label=f"{fluor.name} ({tot:.2f} phot/s tot)"
                )

                em_rate = oc.total_emission_rate(fluor)
                em_rate.plot.line(
                    ax=ab_ax[ch_idx],
                    label=f"{fluor.name} emission",
                    alpha=0.4,
                    linestyle="--",
                )

        # EMISSION PATH --------------------------------------
        for f in oc.filters:
            if f.placement == Placement.EX_PATH:
                continue

            spect = f.spectrum
            if f.placement == Placement.BS_INV:
                spect = spect.inverted()
            em_ax[ch_idx].plot(
                spect.wavelength, spect.intensity, label=f"{f.name}", alpha=0.4
            )

        # detector
        if (detector := sim.detector) and (qe := detector.qe) is not None:
            kwargs = {
                "color": "gray",
                "label": f"{detector.name} QE",
                "linestyle": "--",
                "alpha": 0.4,
            }
            if isinstance(qe, ms.Spectrum):
                em_ax[ch_idx].plot(qe.wavelength, qe.intensity, **kwargs)
            else:
                em_ax[ch_idx].axhline(qe, **kwargs)

        # combined emission/collection
        if ch_em := oc.emission:
            emspec = (ch_em.spectrum * qe).as_xarray()
            emspec.plot.line(ax=em_ax[ch_idx], label="emission", color="k")

            for lbl in sim.sample.labels:
                if fluor := lbl.fluorophore:
                    final = oc.filtered_emission_rate(fluor, detector_qe=qe)
                    final.plot.line(
                        ax=f_ax[ch_idx],
                        label=f"{fluor.name} collection ({final.sum():.2f} phot/s tot)",
                    )

        if legend:
            fp_ax[ch_idx].legend(loc="upper right")
            ex_ax2.legend(loc="upper right")
            ab_ax[ch_idx].legend(loc="upper right")
            f_ax[ch_idx].legend()
            em_ax[ch_idx].legend()
            # oc_ax[ch_idx].legend(loc="right")

        # LABELS --------------------------------------
        ex_ax[ch_idx].set_title(oc.name)
        fp_ax[ch_idx].set_xlabel("")
        ex_ax[ch_idx].set_xlabel("")
        ab_ax[ch_idx].set_xlabel("")
        ab_ax[ch_idx].set_ylabel("[photons/s]")
        em_ax[ch_idx].set_xlabel("")
        f_ax[ch_idx].set_xlabel("wavelength [nm]")
        f_ax[ch_idx].set_ylabel("[photons/s]")

    fp_ax[0].set_xlim(400, 750)  # shared x-axis
    plt.tight_layout()
    plt.show()


# plot_summary(sim)

import xarray as xr


def bin_spectrum_in_region(
    spectrum: xr.DataArray, num_bins: int, intensity_threshold: float = 0.01
) -> xr.DataArray:
    # Extract the coordinates (wavelengths) and values (intensities)
    wavelengths = spectrum.w
    intensities = spectrum.values

    # Filter the spectrum to include only the region of interest (where intensity is significant)
    mask = intensities > intensity_threshold
    filtered_wavelengths = wavelengths[mask]
    filtered_intensities = intensities[mask]

    # Use groupby_bins to bin the data within the filtered region
    filtered_spectrum = xr.DataArray(
        filtered_intensities,
        coords={"w": filtered_wavelengths},
        dims="w",
    )
    binned = filtered_spectrum.groupby_bins("w", bins=num_bins)

    # Calculate the sum of intensities for each bin
    binned_sum = binned.sum(dim="w")

    # Calculate the centroid wavelength for each bin
    centroids = binned.map(lambda x: (x.w * x).sum() / x.sum())

    # Create a new DataArray with the summed intensities and centroid wavelengths
    binned_spectrum = xr.DataArray(
        data=binned_sum.values,
        coords={"w": centroids.values},
        dims="w",
    )

    return binned_spectrum
