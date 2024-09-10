from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from microsim.schema.optical_config.config import LightSource, OpticalConfig
from microsim.schema.optical_config.filter import (
    Bandpass,
    Longpass,
    Placement,
    SpectrumFilter,
    bandpass,
)
from microsim.schema.spectrum import Spectrum

# https://www.chroma.com/products/sets/49000-et-dapi
DAPI = OpticalConfig(
    name="DAPI",
    filters=[
        Bandpass(bandcenter=350, bandwidth=50, placement="EX"),
        Longpass(cuton=400, placement="BS"),
        Bandpass(bandcenter=460, bandwidth=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49001-et-ecfp
ECFP = OpticalConfig(
    name="ECFP",
    filters=[
        Bandpass(bandcenter=436, bandwidth=20, placement="EX"),
        Longpass(cuton=455, placement="BS"),
        Bandpass(bandcenter=480, bandwidth=40, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49002-et-egfp-fitc-cy2
FITC = OpticalConfig(
    name="FITC",
    filters=[
        Bandpass(bandcenter=470, bandwidth=40, placement="EX"),
        Longpass(cuton=495, placement="BS"),
        Bandpass(bandcenter=525, bandwidth=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49003-et-eyfp
EYFP = OpticalConfig(
    name="EYFP",
    filters=[
        Bandpass(bandcenter=500, bandwidth=20, placement="EX"),
        Longpass(cuton=515, placement="BS"),
        Bandpass(bandcenter=535, bandwidth=30, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49004-et-cy3-tritc
TRITC = OpticalConfig(
    name="TRITC",
    filters=[
        Bandpass(bandcenter=545, bandwidth=25, placement="EX"),
        Longpass(cuton=565, placement="BS"),
        Bandpass(bandcenter=605, bandwidth=70, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49005-et-dsred-tritc-cy3
DSRED = OpticalConfig(
    name="DSRED",
    filters=[
        Bandpass(bandcenter=545, bandwidth=30, placement="EX"),
        Longpass(cuton=570, placement="BS"),
        Bandpass(bandcenter=620, bandwidth=60, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49006-et-cy5
CY5 = OpticalConfig(
    name="CY5",
    filters=[
        Bandpass(bandcenter=620, bandwidth=60, placement="EX"),
        Longpass(cuton=660, placement="BS"),
        Bandpass(bandcenter=700, bandwidth=75, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49007-et-cy7
CY7 = OpticalConfig(
    name="CY7",
    filters=[
        Bandpass(bandcenter=710, bandwidth=75, placement="EX"),
        Longpass(cuton=760, placement="BS"),
        Bandpass(bandcenter=810, bandwidth=90, placement="EM"),
    ],
)


# This is just a rough example... needs to be refined
def spectral_detector(
    bins: int,
    min_wave: float,
    max_wave: float,
    lasers: Sequence[int],
    bp_bandwidth: float = 10,
) -> list[OpticalConfig]:
    """Create a spectral detector with a given number of bins and lasers.

    Parameters
    ----------
    bins : int
        Number of bins to use.
    min_wave : float
        Minimum wavelength to consider.
    max_wave : float
        Maximum wavelength to consider.
    lasers : Sequence[int]
        List of lasers to use.
    bp_bandwidth : float, optional
        Bandwidth of the bandpass filter, by default 10.

    Returns
    -------
    list[ms.OpticalConfig]
        List of optical configurations.

    Examples
    --------
    Create a spectral detector with 8 bins, wavelengths between 480 and 600 nm, and
    lasers at 488, 515, and 561 nm.

    >>> spectral_detector(8, 480, 600, [488, 515, 561])
    """
    lights = [LightSource.laser(laser) for laser in lasers]
    waves = np.arange(min_wave - 100, max_wave + 100, 1)

    # create fake bandpass ... could also be something like 80/20
    laser0, *rest = lasers
    bp = bandpass(waves, center=laser0, bandwidth=bp_bandwidth, transmission=1)
    for laser in rest:
        bp = bp + bandpass(waves, center=laser, bandwidth=bp_bandwidth)
    bp = SpectrumFilter(
        transmission=Spectrum(wavelength=waves, intensity=1 - bp),
        placement=Placement.BS,
    )

    configs: list[OpticalConfig] = []
    edges = np.linspace(min_wave, max_wave, bins + 1)
    for i, (low, high) in enumerate(pairwise(edges)):
        mask = (waves >= low) & (waves <= high)
        f = SpectrumFilter(
            transmission=Spectrum(wavelength=waves, intensity=mask),
            placement=Placement.EM_PATH,
        )
        oc = OpticalConfig(
            name=f"Channel {i}",
            lights=lights,
            filters=[bp, f],
        )
        configs.append(oc)
    return configs
