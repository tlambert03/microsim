"""Point-scanning detectors: photomultiplier tube (PMT) and hybrid detector (HyD).

A point scanner builds the image one pixel at a time, so `exposure_ms` is the pixel
dwell time, typically 0.0005-0.01 ms (0.5-10 µs).  These detectors have no pixels
that store charge, so they have no `full_well`, and their analog gain is set with
`electrons_per_adu`.  See `_detector.py` for the simulation steps shared by all
detectors.
"""

import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy import signal

from ._detector import PositiveFloat, _Detector

if TYPE_CHECKING:
    from typing import Self

    import numpy.typing as npt

    from microsim.schema.backend import NumpyAPI

ELEMENTARY_CHARGE = 1.602176634e-19  # C


class PMT(_Detector):
    """Photomultiplier tube, read out in analog (current-integrating) mode.

    A photon hitting the photocathode releases a photoelectron (with probability
    `qe`).  The photoelectron is accelerated onto a chain of 8-12 dynodes, each of
    which releases ~3-6 secondary electrons per incoming electron, for a total gain
    of 1e4-1e7 electrons per photoelectron at the anode.  The anode current is
    integrated over the pixel dwell time, amplified, and digitized.

    The gain is set by the high voltage (HV) across the tube, which is what
    microscope software calls the PMT "gain", "HV", or "master gain".  Each dynode
    multiplies by `delta ~ V**k` (k = 0.7-0.8), so the total gain follows a power law
    (Hamamatsu PMT Handbook v4, Eq. 4-9):

        gain = ref_gain * (hv / ref_hv) ** hv_exponent

    `hv` is in whatever units `ref_hv` is.  The presets describe Hamamatsu
    photosensor modules (as used in many confocals), whose HV is set by a 0.5-0.9 V
    control voltage.  For a bare tube, use volts (e.g. `ref_hv=1000`).  Experts can
    set `hv_gain` directly instead.

    Each dynode's yield is random, which adds gain noise.  With `n` equal stages of
    mean yield `d = gain ** (1/n)`, the excess noise factor is
    `F = 1 + 1/d + 1/d**2 + ... + 1/d**n`, about `d / (d - 1)` (Handbook Eq. 4-25).
    Most of it comes from the first dynode, where the fewest electrons are involved.
    `F` is ~1.3 at a gain of 5e5, and rises as the HV is lowered (1.5 at 2e4).  Real
    modules often run the first dynode at a higher voltage, which lowers `F` a bit;
    that is not modeled.

    The amplifier adds a noise current, so the noise charge integrated over a pixel
    grows as `sqrt(dwell)`, while the signal grows as `dwell`.  It is negligible
    unless the HV (and so the gain) is low.

    The anode current cannot rise without limit.  With continuous light the limit
    comes from the voltage divider that sets the dynode voltages: anode current is
    drawn from it, the last dynodes lose voltage, and the gain drops (Handbook
    5.1.3).  Modules with an active divider stay linear almost up to their rated
    maximum average output current, then drop sharply (Hamamatsu H10720 datasheet,
    Fig. 8).  Capacitors across the last dynodes supply short bursts of current, so
    the divider responds to the anode current averaged over its RC time constant
    (`divider_time_constant_ms`), not to single pixels: a lone bright pixel is fine,
    but a bright region dims itself and the scan lines after it.  This is modeled as
    a gain factor `1 / (1 + (I_avg / I_max) ** 6) ** (1/6)`, where `I_avg` is a
    running average of the anode current in scan order (rows left to right, top to
    bottom, ignoring flyback): ~1% low at `0.6 I_max`, 11% low at `I_max`.
    Lowering the HV raises the light level at which this happens.

    Known omissions:

    - The divider's response is a single time constant, and starts each image from
      the image's mean current (as when scanning repeatedly), ignoring any recovery
      during frame flyback.
    - A few % rise in gain before saturation ("over-linearity", Handbook Fig. 5-5).
    - Space charge saturation, which limits *peak* current (mA) and matters only for
      pulsed excitation, which microsim does not model.
    - Afterpulses (ion feedback a few hundred ns to µs after a photoelectron): a
      few % of extra delayed signal, with no published rates.
    - Temperature dependence, drift, and hysteresis of the gain (~1%).

    Attributes
    ----------
    hv : float, optional
        HV (or control voltage) setting.  Defaults to `ref_hv`.
    hv_range : tuple[float, float], optional
        Usable range of `hv`.  A warning is issued outside it.  The default is the
        H7422-40's: 0.5 V (bottom of the recommended range) to 0.9 V (maximum
        rating).
    ref_hv, ref_gain, hv_exponent : float
        Gain curve of the tube: `ref_gain` at `ref_hv`, rising as `hv**hv_exponent`.
        The defaults are read from the Hamamatsu H7422-40 (GaAsP) datasheet: gain
        5e5 at 0.8 V, and 2e4-1e6 over 0.5-0.9 V.
    hv_gain : float, optional
        Mean number of anode electrons per photoelectron.  Overrides `hv`.
    dynode_stages : int
        Number of dynodes.  Only used for the excess noise factor.  The default (9)
        matches `hv_exponent / k` for the presets, with `k = 0.75`.
    digital_gain : float
        Multiplies the output after digitization, before `offset` (as in Zeiss ZEN).
        Unlike `hv`, it amplifies signal and noise equally, and leaves gaps between
        the gray values used.
    electrons_per_adu : float
        Anode electrons per gray value (the amplifier/digitizer gain), before
        `digital_gain`.  The default gives 5 gray values per photoelectron at the
        default gain.
    read_noise : float
        Amplifier noise, in anode electrons rms, for a 1 µs dwell.  Scales with
        `sqrt(dwell)`.  The default is an estimate, not a published figure: an
        input noise current of ~1 pA/sqrt(Hz) integrated over 1 µs.
    dark_current : float
        Dark counts (thermionic emission from the photocathode) per second.
    max_output_current_ua : float, optional
        Rated maximum average anode current (µA), where the output saturates.
        Hamamatsu H7422: 2 µA for GaAsP, 100 µA for multialkali.  `None` for no
        limit (other than the ADC).
    divider_time_constant_ms : float
        Time (ms) over which the voltage divider averages the anode current.  Not
        published for any module; the default (3 ms) is the RC of the example
        divider in Handbook Fig. 5-30 (300-500 kOhm per stage, 0.01 µF).
    """

    camera_type: Literal["PMT"] = "PMT"
    hv: Annotated[float, Field(gt=0)] | None = None
    hv_range: tuple[float, float] | None = (0.5, 0.9)
    ref_hv: Annotated[float, Field(gt=0)] = 0.8
    ref_gain: Annotated[float, Field(ge=1)] = 5e5
    hv_exponent: Annotated[float, Field(gt=0)] = 6.7
    hv_gain: Annotated[float, Field(ge=1)] | None = None
    dynode_stages: Annotated[int, Field(ge=1)] = 9
    digital_gain: Annotated[float, Field(gt=0)] = 1
    electrons_per_adu: Annotated[float, Field(gt=0)] = 1e5
    read_noise: PositiveFloat = 5e3
    dark_current: PositiveFloat = 0
    offset: int = 0
    max_output_current_ua: Annotated[float, Field(gt=0)] | None = None
    divider_time_constant_ms: Annotated[float, Field(gt=0)] = 3

    @model_validator(mode="after")
    def _check_hv(self) -> "Self":
        if self.hv is not None and self.hv_gain is not None:
            raise ValueError("Specify either `hv` or `hv_gain`, not both.")
        if self.hv is not None and self.hv_range is not None:
            lo, hi = self.hv_range
            if not lo <= self.hv <= hi:
                warnings.warn(
                    f"hv={self.hv} is outside the usable range {self.hv_range} of "
                    f"this PMT.",
                    stacklevel=2,
                )
        return self

    @property
    def multiplication_gain(self) -> float:
        if self.hv_gain is not None:
            return self.hv_gain
        hv = self.ref_hv if self.hv is None else self.hv
        return float(self.ref_gain * (hv / self.ref_hv) ** self.hv_exponent)

    @property
    def excess_noise_factor(self) -> float:
        delta = self.multiplication_gain ** (1 / self.dynode_stages)
        return float(sum(delta**-i for i in range(self.dynode_stages + 1)))

    def _read_noise_per_pass(self, dwell_s: Any) -> Any:
        return self.read_noise * (dwell_s / 1e-6) ** 0.5

    def _after_multiplication(
        self, electrons: "npt.NDArray", time_s: Any, xp: "NumpyAPI"
    ) -> "npt.NDArray":
        if self.max_output_current_ua is not None:
            current = electrons * ELEMENTARY_CHARGE / time_s
            tau_s = self.divider_time_constant_ms * 1e-3
            recent = _per_channel(_raster_running_mean, current, time_s, tau_s)
            ratio = recent / (self.max_output_current_ua * 1e-6)
            electrons = electrons / (1 + ratio**6) ** (1 / 6)
        return electrons

    def _digital_gain(self) -> float:
        return self.digital_gain


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


def _per_channel(func: Any, x: "npt.NDArray", dt: Any, *args: Any) -> "npt.NDArray":
    """Call `func(x, dt, *args)`, per channel (first axis) if `dt` is per channel."""
    x, dt = np.asarray(x, dtype=float), np.asarray(dt, dtype=float)
    if dt.size == 1:
        return func(x, float(dt), *args)  # type: ignore[no-any-return]
    return np.stack(
        [func(xc, float(d), *args) for d, xc in zip(dt.ravel(), x, strict=True)]
    )


def _raster_running_mean(x: "npt.NDArray", dt: float, tau: float) -> "npt.NDArray":
    """Exponential running mean over the last two axes (Y, X), in raster order.

    `dt` is the time per pixel, and `tau` the averaging time constant.  Each image
    starts from its own mean, as if it had been scanned before.
    """
    decay = np.exp(-dt / tau)
    flat = x.reshape(*x.shape[:-2], -1)
    zi = decay * flat.mean(axis=-1, keepdims=True)  # steady state at the mean
    out, _ = signal.lfilter([1 - decay], [1, -decay], flat, axis=-1, zi=zi)
    return out.reshape(x.shape)  # type: ignore[no-any-return]
