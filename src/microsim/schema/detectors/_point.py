"""Point-scanning (single-element) detectors: PMT and hybrid photodetector (HyD).

For these detectors, `exposure_ms` (as passed by `Simulation.digital_image`) is
interpreted as the pixel dwell time.  Realistic values are on the order of
0.0005 - 0.01 ms (0.5 - 10 µs).  Camera-specific fields inherited from `_Camera`
(`full_well`, `serial_reg_full_well`, `clock_induced_charge`) are ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import numpy.typing as npt
from annotated_types import Ge
from pydantic import Field
from scipy import stats

from microsim.schema.backend import NumpyAPI

from ._camera import _Camera

if TYPE_CHECKING:
    from microsim._data_array import ArrayProtocol, xrDataArray

PositiveFloat = Annotated[float, Ge(0)]


class _PointDetector(_Camera):
    """Base class for point-scanning detectors.

    Attributes
    ----------
    averaging : int
        Number of line/frame averages.  Photons are integrated for
        `exposure_ms * averaging`, and the result is divided by `averaging`.
    dark_current : float
        Dark count rate, in counts per second.
    offset : int
        Offset added to the output, in gray values.
    read_noise : float
        Electronic noise, in gray values (rms) per pass.
    """

    averaging: int = Field(1, ge=1)
    dark_current: PositiveFloat = Field(0, description="counts/sec")
    offset: int = 0
    read_noise: PositiveFloat = 0

    def _mean_photoelectrons(
        self, photons_per_second: xrDataArray, exposure_ms: float | xrDataArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Return (mean photoelectrons, total integration time in s) per pixel."""
        # NOTE: QE is already applied upstream in `filtered_emission_rate`
        t_s = exposure_ms / 1000 * self.averaging
        flux = photons_per_second.clip(min=0) + self.dark_current
        lam = flux * t_s
        t_s = t_s + 0 * lam  # broadcast (e.g. per-channel exposure) to image shape
        return lam.data, t_s.data

    def _to_gray(self, values: npt.NDArray, xp: NumpyAPI) -> npt.NDArray:
        """Average, add electronic noise and offset, round, and clip to bit depth."""
        if self.read_noise > 0:
            noise = self.read_noise * np.sqrt(self.averaging)
            values = xp.norm_rvs(values, noise)  # type: ignore[assignment]
        gray = xp.round(values / self.averaging + self.offset)
        gray = xp.clip(gray, 0, self.max_intensity)
        return gray.astype(  # type: ignore[no-any-return]
            "uint16" if self.bit_depth <= 16 else "uint32"
        )


class PMT(_PointDetector):
    """Analog (integrating) photomultiplier tube.

    Photoelectrons are Poisson distributed.  Each photoelectron is multiplied by a
    gamma-distributed gain with mean `gain` and excess noise factor `enf` (the sum
    of N such gammas is itself gamma: Gamma(N / (F-1), gain * (F-1))).  Electronic
    noise and offset are then added before quantization.

    Attributes
    ----------
    gain : float
        Mean gray values per photoelectron.  Stands in for the PMT high voltage
        ("gain" / "master gain" in acquisition software).
    enf : float
        Excess noise factor of the dynode chain (F = 1 + Var(g) / mean(g)^2).
        ~1.2 - 1.5 for typical PMTs; 2 for an exponential single-photon response.
    """

    camera_type: Literal["PMT"] = "PMT"
    gain: PositiveFloat = 10
    enf: float = Field(1.3, gt=1)
    read_noise: PositiveFloat = 1

    def simulate(
        self,
        photons_per_second: xrDataArray,
        exposure_ms: float | xrDataArray = 0.002,
        binning: int = 1,
        add_poisson: bool = True,
        xp: NumpyAPI | None = None,
    ) -> ArrayProtocol:
        xp = NumpyAPI.create(xp)
        lam, _ = self._mean_photoelectrons(photons_per_second, exposure_ms)
        n_pe = np.asarray(xp.poisson_rvs(lam, shape=lam.shape) if add_poisson else lam)
        k = 1 / (self.enf - 1)
        theta = self.gain * (self.enf - 1)
        charge = stats.gamma.rvs(np.maximum(n_pe, 1e-9) * k, scale=theta)
        charge = np.where(n_pe > 0, charge, 0)
        return self._to_gray(xp.asarray(charge), xp)


class HyD(_PointDetector):
    """Hybrid photodetector (e.g. Leica HyD) operated as a photon counter.

    The first-stage (electron bombardment) gain of ~1500 makes gain noise
    negligible, so the output is a Poisson photon count, reduced at high count
    rates by a non-paralyzable dead time: m = n / (1 + n * tau).

    Attributes
    ----------
    mode : {"counting", "standard"}
        "counting" returns raw photon counts.  "standard" scales counts by `gain`
        (Leica's "Standard" mode; its exact gain mapping and linearization are
        undisclosed, so this is a plain multiplication).
    gain : float
        Gray values per counted photon in "standard" mode.  Ignored in "counting".
    dead_time_ns : float
        Dead time of detector + electronics, in ns (Leica: < 1.5 ns).
    """

    camera_type: Literal["HyD"] = "HyD"
    mode: Literal["counting", "standard"] = "counting"
    gain: PositiveFloat = 1
    dead_time_ns: PositiveFloat = 1.5

    def simulate(
        self,
        photons_per_second: xrDataArray,
        exposure_ms: float | xrDataArray = 0.002,
        binning: int = 1,
        add_poisson: bool = True,
        xp: NumpyAPI | None = None,
    ) -> ArrayProtocol:
        xp = NumpyAPI.create(xp)
        lam, t_s = self._mean_photoelectrons(photons_per_second, exposure_ms)
        # non-paralyzable dead time applied to the mean rate
        lam = lam / (1 + (lam / t_s) * self.dead_time_ns * 1e-9)
        counts = xp.poisson_rvs(lam, shape=lam.shape) if add_poisson else lam
        if self.mode == "standard":
            counts = counts * self.gain
        return self._to_gray(counts, xp)
