from matplotlib import pyplot as plt

from microsim import schema as ms
from microsim.schema.optical_config import Placement

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.04, 0.02, 0.02)),
    output_space={"downscale": 4},
    sample=ms.Sample(
        labels=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(), fluorophore="mStayGold"
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(), fluorophore="mScarlet3"
            ),
        ]
    ),
    channels=[
        ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Green"),
        ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Red"),
    ],
    modality=ms.Confocal(),
    detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12),
)


def plot_summary(sim: ms.Simulation) -> None:
    fig, ax = plt.subplots(5, len(sim.channels), figsize=(18, 10), sharex=True)
    # optical configs
    fp_ax, ex_ax, ab_ax, em_ax, f_ax = ax
    if len(sim.channels) == 1:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = [fp_ax], [ex_ax], [ab_ax], [em_ax], [f_ax]
    for ch_idx, oc in enumerate(sim.channels):
        # FLUOROPHORES --------------------------------------
        for lbl in sim.sample.labels:
            if fluor := lbl.fluorophore:
                ex = fluor.absorption_cross_section
                ex.plot(ax=fp_ax[ch_idx], label=f"{fluor.name}")

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
            ex = light.spectrum
            ex_ax2.plot(ex.wavelength, ex.intensity, label=f"{light.name}", alpha=0.4)

        # combined illumination
        full = oc.illumination_flux_density
        full.plot(ax=ex_ax[ch_idx], label="flux density", color="k")

        # ABSORPTION/EMISSION RATES --------------------------------------
        for lbl in sim.sample.labels:
            if fluor := lbl.fluorophore:
                rate = oc.absorption_rate(fluor)
                tot = rate.sum()
                rate.plot(
                    ax=ab_ax[ch_idx], label=f"{fluor.name} ({tot:.2f} phot/s tot)"
                )

                em_rate = oc.total_emission_rate(fluor)
                em_rate.plot(
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
        if sim.detector and (qe := sim.detector.qe) is not None:
            if isinstance(qe, ms.Spectrum):
                em_ax[ch_idx].plot(
                    qe.wavelength, qe.intensity, label=f"{f.name}", alpha=0.4
                )
            else:
                em_ax[ch_idx].axhline(
                    qe, color="gray", linestyle="--", alpha=0.4, label="QE"
                )

        # combined emission/collection
        if ch_em := oc.emission:
            emspec = ch_em.spectrum.as_xarray()
            emspec = emspec * qe
            emspec.plot(ax=em_ax[ch_idx], label="emission", color="k")

            for lbl in sim.sample.labels:
                if fluor := lbl.fluorophore:
                    combined = oc.filtered_emission_rate(fluor)
                    combined = combined * qe
                    tot = combined.sum()
                    combined.plot(
                        ax=f_ax[ch_idx],
                        label=f"{fluor.name} collection ({tot:.2f} phot/s tot)",
                    )

        # LABELS --------------------------------------
        fp_ax[ch_idx].set_xlabel("")
        fp_ax[ch_idx].legend(loc="upper right")
        fp_ax[ch_idx].set_title(oc.name + " channel")
        ex_ax2.legend(loc="upper right")
        # oc_ax[ch_idx].legend(loc="right")
        ex_ax[ch_idx].set_xlabel("")
        ab_ax[ch_idx].legend(loc="upper right")
        ab_ax[ch_idx].set_xlabel("")
        ab_ax[ch_idx].set_ylabel("Abs/Em Rate [photons/s]")
        f_ax[ch_idx].legend()
        em_ax[ch_idx].legend()
        em_ax[ch_idx].set_xlabel("")
        f_ax[ch_idx].set_xlabel("wavelength [nm]")

    fp_ax[0].set_xlim(400, 750)  # shared x-axis
    plt.tight_layout()
    plt.show()


plot_summary(sim)
