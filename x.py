from collections.abc import Sequence
from itertools import pairwise

import numpy as np

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
                distribution=ms.MatsLines(density=0.5), fluorophore="mEGFP"
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=1.0), fluorophore="mVenus"
            ),
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=1.5), fluorophore="mCherry"
            ),
        ]
    ),
    channels=[
        ms.OpticalConfig.from_fpbase("wKqWbg", "Widefield Green"),
        ms.OpticalConfig.from_fpbase("wKqWbg", "Widefield Red"),
    ],
    # channels=spectral_detector(8, 480, 600, [488, 515, 561]),
    modality=ms.Confocal(),
    detector=ICX285,
)


sim.plot()

# import xarray as xr
