"""Point-scanning detectors: photomultiplier tube (PMT) and hybrid detector (HyD).

A point scanner builds the image one pixel at a time, so `exposure_ms` is the pixel
dwell time, typically 0.0005-0.01 ms (0.5-10 µs).  These detectors have no pixels
that store charge, so they have no `full_well`, and their analog gain is set with
`electrons_per_adu`.  See `_detector.py` for the simulation steps shared by all
detectors.
"""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from ._detector import PositiveFloat, _Detector

if TYPE_CHECKING:
    import numpy.typing as npt


class PMT(_Detector):
    """Photomultiplier tube, read out in analog (current-integrating) mode.

    A photon hitting the photocathode releases a photoelectron (with probability
    `qe`).  The photoelectron is accelerated onto a chain of 8-12 dynodes, each of
    which releases ~3-6 secondary electrons per incoming electron, for a total gain
    of 1e5-1e7 electrons per photoelectron at the anode.  The gain rises steeply with
    the high voltage (HV) across the tube, which is what microscope software calls
    the PMT "gain" or "master gain".  The anode current is integrated over the pixel
    dwell time, amplified, and digitized.

    Gain noise comes mostly from the first dynode, where the fewest electrons are
    involved: with a mean of `d` secondaries, `F` is about `d / (d - 1)`, i.e.
    1.2-1.5.  `enf` is fixed here; on a real tube it rises as the HV is lowered.

    Attributes
    ----------
    hv_gain : float
        Mean number of electrons at the anode per photoelectron (set by the HV).
    enf : float
        Excess noise factor of the dynode chain.
    electrons_per_adu : float
        Anode electrons per gray value (the amplifier/digitizer gain).  The default,
        with the default `hv_gain`, gives 10 gray values per photoelectron.
    read_noise : float
        Amplifier noise, in anode electrons rms.
    dark_current : float
        Dark counts (thermionic emission from the photocathode) per second.
    """

    camera_type: Literal["PMT"] = "PMT"
    hv_gain: Annotated[float, Field(ge=1)] = 1e6
    enf: Annotated[float, Field(ge=1)] = 1.3
    electrons_per_adu: Annotated[float, Field(gt=0)] = 1e5
    read_noise: PositiveFloat = 1e5
    dark_current: PositiveFloat = 0
    offset: int = 0

    @property
    def multiplication_gain(self) -> float:
        return self.hv_gain

    @property
    def excess_noise_factor(self) -> float:
        return self.enf


class HyD(_Detector):
    """Hybrid photodetector (e.g. Leica HyD), read out by photon counting.

    A photocathode (as in a PMT) releases a photoelectron, which is accelerated by
    ~8 kV straight into an avalanche diode.  The impact alone creates ~1500
    electron-hole pairs, and the avalanche multiplies them ~100x more.  Because the
    first stage is so large, every photon gives nearly the same pulse (~3% spread),
    so pulses are counted rather than integrated.  The output is a photon count: no
    gain noise (`F = 1`), and no read noise.

    Each pulse briefly blinds the detector (`dead_time_ns`), so at high count rates
    some photons are missed: a true rate `n` is recorded as `n / (1 + n * tau)`.  At
    40 Mcps and 1.5 ns, that loses ~6%.  The dead time also spaces counts more
    regularly than random arrivals, so a mean count `m` has variance
    `m / (1 + n * tau)**2` rather than `m`.

    In Leica's "Standard" mode the count is multiplied by a gain; that is
    `electrons_per_adu = 1 / gain` here.  The linearization Leica applies in that
    mode is not published, and is not modeled.

    Attributes
    ----------
    dead_time_ns : float
        Time after each pulse during which no other photon is counted (Leica: < 1.5
        ns for the detector and electronics).
    electrons_per_adu : float
        Photons per gray value.  1 (the default) outputs raw photon counts.
    dark_current : float
        Dark counts per second.
    """

    camera_type: Literal["HyD"] = "HyD"
    dead_time_ns: PositiveFloat = 1.5
    electrons_per_adu: Annotated[float, Field(gt=0)] = 1
    read_noise: PositiveFloat = 0
    dark_current: PositiveFloat = 0
    offset: int = 0

    def _counting_efficiency(
        self, mean_events: "npt.NDArray", time_s: Any
    ) -> "npt.NDArray":
        # non-paralyzable dead time: m = n / (1 + n * tau), applied to the mean
        rate = mean_events / time_s
        return 1 / (1 + rate * self.dead_time_ns * 1e-9)  # type: ignore[no-any-return]
