"""Base detector model.

Every detector converts an image of photon flux (photons/s per pixel) into gray
values with the same sequence of steps.  Detectors differ only in their parameters,
and in which steps do nothing for them:

1. **Expected signal.** Mean photoelectrons per pixel: flux x integration time, plus
   dark current x integration time (plus clock-induced charge for cameras).  The
   integration time is `exposure_ms` (pixel dwell time for point scanners) times
   `averaging`.  Photon counters lose some events at high count rates (dead time),
   which also makes their counts less noisy than Poisson.
   QE is not applied here: it is applied (per wavelength) to the emission rates
   upstream, in `Simulation.filtered_emission_rates`.
2. **Shot noise.** Photon arrival and thermal generation are Poisson processes, so
   the number of photoelectrons in each pixel is Poisson distributed (less for
   photon counters at high count rates, where dead time spaces counts out).
3. **Full well.** A camera pixel cannot hold more than `full_well` electrons.
   Charge binning (CCD) happens here, before readout.
4. **Multiplication.** Some detectors amplify each electron before readout (the EM
   register of an EMCCD, the dynodes of a PMT).  The gain `g` of each electron is
   random, which adds noise on top of shot noise.  This is summarized by the
   *excess noise factor*, `F = <g^2> / <g>^2 = 1 + var(g) / mean(g)^2`: the output
   variance is `F` times what a noiseless gain would give, so `SNR = sqrt(n / F)`.
   Gain noise acts like dividing the QE by `F`.  Detectors without multiplication
   have gain 1 and `F = 1`.
5. **Readout.** The amplifier adds Gaussian read noise (`read_noise`, electrons
   rms, once per frame or pass).  The ADC then converts electrons to gray values
   (the analog gain, `electrons_per_adu`), averages over `averaging` passes, rounds
   (which adds quantization noise), and clips at the top of the range of
   `bit_depth`.  Then any digital gain is applied, `offset` is added, and the
   result is clipped to the range again.  Digital binning (CMOS) happens after this
   step.

Read noise is modeled as a single term referred to the input, so it does not
change with analog gain or readout rate.
"""

from typing import TYPE_CHECKING, Annotated

import numpy as np
import numpy.typing as npt
from annotated_types import Ge, Interval
from pydantic import Field, model_validator
from scipy import stats

from microsim._data_array import DataArray, xrDataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.spectrum import Spectrum

if TYPE_CHECKING:
    from typing import Any, Self

    from microsim._data_array import ArrayProtocol

PositiveFloat = Annotated[float, Ge(0)]
PositiveInt = Annotated[int, Ge(0)]


def apply_multiplication_gain(
    electrons: npt.NDArray, gain: float, enf: float
) -> npt.NDArray:
    """Multiply each electron by a random gain with mean `gain` and noise factor `enf`.

    Each electron's gain is drawn from a gamma distribution with mean `gain` and
    variance `(enf - 1) * gain**2`.  A sum of `n` such gammas is itself gamma
    distributed, `Gamma(shape=n / (enf - 1), scale=gain * (enf - 1))`, so the whole
    pixel is drawn at once.  This matches the mean and variance of the true gain
    distribution, and is exact for an EMCCD at high gain, where each electron's gain
    is exponential (`enf` = 2).  At low gain and few input electrons the shape of the
    output distribution is only approximate (e.g. a real EM register rarely outputs
    fewer electrons than it receives, but at `gain` = 2 this outputs 0 for ~9% of
    single electrons), though its mean and variance remain correct.
    """
    electrons = np.asarray(electrons)
    if enf - 1 < 1e-9:  # noiseless gain
        return np.round(electrons * gain)
    out = np.zeros(electrons.shape)
    mask = electrons > 0  # gamma shape must be > 0; empty pixels stay empty
    out[mask] = stats.gamma.rvs(electrons[mask] / (enf - 1), scale=gain * (enf - 1))
    return np.round(out)


def _per_channel(value: "Any", ndim: int) -> "Any":
    """Reshape a per-channel value (DataArray over C) to broadcast against images."""
    if isinstance(value, float | int):
        return value
    arr = np.asarray(value)
    return arr.reshape(arr.shape + (1,) * (ndim - arr.ndim))


class _Detector(SimBaseModel):
    """Base detector model (see module docstring for the simulation steps).

    Attributes
    ----------
    camera_type : str
        Type of detector, for discriminated union.
    name : str
        A descriptive name for the detector.  Not used internally.
    qe : float | Spectrum
        Quantum efficiency, from 0-1: the probability that a photon creates a
        photoelectron.  A float is constant across wavelengths.
    dark_current : float
        Thermally generated electrons (or dark counts) per pixel per second.
    full_well : int, optional
        Maximum number of electrons a pixel can hold.  `None` for detectors without
        pixels that store charge (e.g. PMTs).
    read_noise : float
        Read noise, in electrons rms, added once per frame (or pass).
    relative_gain : float, optional
        Analog (pre-amplifier) gain, relative to the setting at which `full_well`
        electrons map to the maximum gray value.  This is what microscope software
        usually calls a gain of 1.  Requires `full_well`.  Mutually exclusive with
        `electrons_per_adu`.  Defaults to 1 if neither is given.
    electrons_per_adu : float, optional
        Analog gain as electrons per gray value, as listed on camera datasheets.
        Mutually exclusive with `relative_gain`.
    offset : int
        Constant added to every pixel, in gray values.  Cameras use it so that read
        noise is not clipped at zero.  On a point scanner it is a user setting, often
        lowered to clip background to zero.
    bit_depth : int
        ADC bit depth.  Gray values are clipped to `[0, 2**bit_depth - 1]`.
    averaging : int
        Number of frames (or line/frame passes, for point scanners) averaged into
        each image.  Signal and dark current are integrated for `averaging` times
        the exposure, read noise is added once per pass, and the result is divided
        by `averaging`.  This lowers noise without changing brightness.
    """

    camera_type: str = "generic"
    name: str = ""
    qe: Annotated[float, Interval(ge=0, le=1)] | Spectrum = 1
    dark_current: PositiveFloat = Field(0.001, description="e/pix/sec")
    full_well: int | None = None
    read_noise: PositiveFloat = 6  # TODO: accept map of readout rate -> noise?
    relative_gain: Annotated[float, Field(gt=0)] | None = None
    electrons_per_adu: Annotated[float, Field(gt=0)] | None = None
    offset: int = 100
    bit_depth: PositiveInt = 12
    averaging: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def _check_analog_gain(self) -> "Self":
        if self.full_well is None and (
            self.relative_gain is not None or self.electrons_per_adu is None
        ):
            raise ValueError(
                f"{type(self).__name__} has no `full_well`, so `relative_gain` is "
                "undefined. Specify `electrons_per_adu` instead."
            )
        if self.relative_gain is not None and self.electrons_per_adu is not None:
            raise ValueError(
                "Specify either `relative_gain` or `electrons_per_adu`, not both."
            )
        return self

    @property
    def conversion_factor(self) -> float:
        """Electrons per gray value, from `electrons_per_adu` or `relative_gain`."""
        if self.electrons_per_adu is not None:
            return self.electrons_per_adu
        assert self.full_well is not None  # checked by the validator
        return self.full_well / (self.max_intensity * (self.relative_gain or 1))

    @property
    def max_intensity(self) -> int:
        return int(2**self.bit_depth - 1)

    # Hooks for the steps that only some detectors have.  The defaults do nothing.

    @property
    def multiplication_gain(self) -> float:
        """Mean gain applied to each electron before readout (1 if none)."""
        return 1

    @property
    def excess_noise_factor(self) -> float:
        """Excess noise factor of the multiplication step (1 if none)."""
        return 1

    def _dark_electrons_per_frame(self) -> float:
        """Mean electrons added once per frame, regardless of exposure (e.g. CIC)."""
        return 0

    def _digital_gain(self) -> float:
        """Factor applied to the digitized signal (before the offset)."""
        return 1

    def _counting_efficiency(
        self, mean_events: npt.NDArray, time_s: "Any"
    ) -> "npt.NDArray | None":
        """Fraction of `mean_events` (in `time_s`) that are recorded, or None if all."""
        return None

    def _clip_after_multiplication(
        self, electrons: npt.NDArray, xp: NumpyAPI
    ) -> npt.NDArray:
        return electrons

    def apply_pre_quantization_binning(
        self, total_electrons: npt.NDArray, binning: int
    ) -> npt.NDArray:
        return total_electrons

    def apply_post_quantization_binning(
        self, gray_values: npt.NDArray, binning: int
    ) -> npt.NDArray:
        return gray_values

    def simulate(
        self,
        photons_per_second: "xrDataArray",
        exposure_ms: "float | xrDataArray" = 100,
        binning: int = 1,
        add_poisson: bool = True,
        xp: "NumpyAPI | None" = None,
    ) -> "ArrayProtocol":
        xp = NumpyAPI.create(xp)
        n_avg = self.averaging
        exposure_s = exposure_ms / 1000 * n_avg  # total integration time

        # 1. expected signal
        # NOTE: QE is applied upstream (Simulation.filtered_emission_rates), not here.
        # This is only correct if the optical image was computed with this detector;
        # see https://github.com/tlambert03/microsim/issues/152
        incident_photons = xp.maximum((photons_per_second * exposure_s).data, 0)
        ndim = incident_photons.ndim
        # dark current and clock-induced charge (mean electrons per pixel)
        avg_dark_e = _per_channel(
            self.dark_current * exposure_s + self._dark_electrons_per_frame() * n_avg,
            ndim,
        )
        # events lost at high count rates (photon counters)
        efficiency = self._counting_efficiency(
            incident_photons + avg_dark_e, _per_channel(exposure_s, ndim)
        )
        if efficiency is not None:
            incident_photons = incident_photons * efficiency
            avg_dark_e = avg_dark_e * efficiency

        # 2. shot noise
        if add_poisson:
            electrons = xp.poisson_rvs(incident_photons, shape=incident_photons.shape)
        else:
            electrons = incident_photons
        electrons = electrons + xp.poisson_rvs(avg_dark_e, shape=electrons.shape)
        if efficiency is not None:
            # dead time also makes counts more regular than Poisson: a renewal process
            # with mean m has variance m * efficiency**2.  Shrink each deviation.
            # (Valid when the dwell is much longer than the dead time; rounding makes
            # it approximate at a count or two per pixel.)
            mean = incident_photons + avg_dark_e
            electrons = xp.round(mean + (electrons - mean) * efficiency)

        # 3. full well, and charge binning
        if self.full_well is not None:
            electrons = xp.minimum(electrons, self.full_well * n_avg)
        if binning > 1:
            electrons = self.apply_pre_quantization_binning(electrons, binning)

        # 4. multiplication
        if self.multiplication_gain != 1:
            electrons = apply_multiplication_gain(
                electrons, self.multiplication_gain, self.excess_noise_factor
            )
        electrons = self._clip_after_multiplication(electrons, xp)

        # 5. readout: read noise, ADC conversion, averaging, offset, binning, clipping
        voltage = xp.norm_rvs(electrons, self.read_noise * n_avg**0.5)  # in electrons
        adu = self.conversion_factor * n_avg  # electrons per gray value, averaged
        signal = xp.minimum(xp.round(voltage / adu), self.max_intensity)  # type: ignore[operator]
        signal = xp.round(signal * self._digital_gain())
        gray = xp.maximum(signal + self.offset, 0)
        if binning > 1:
            gray = self.apply_post_quantization_binning(gray, binning)
        gray = xp.minimum(gray, self.max_intensity)
        if self.bit_depth > 16:
            return gray.astype("uint32")  # type: ignore[no-any-return]
        if self.bit_depth > 8:
            return gray.astype("uint16")  # type: ignore[no-any-return]
        return gray.astype("uint8")  # type: ignore[no-any-return]

    def render(
        self,
        image: xrDataArray,
        exposure_ms: float | xrDataArray = 100,
        binning: int = 1,
        add_poisson: bool = True,
        xp: NumpyAPI | None = None,
    ) -> xrDataArray:
        """Simulate imaging process.

        Parameters
        ----------
        image : DataArray
            array where each element represents photons / second
        exposure_ms : float, optional
            Exposure time (pixel dwell time for point scanners) in milliseconds, by
            default 100
        binning : int, optional
            Binning to apply to the last two (Y, X) dimensions, by default 1
        add_poisson : bool, optional
            Whether to add poisson noise, by default True
        xp: NumpyAPI | None
            Numpy API backend
        """
        new_data = self.simulate(
            photons_per_second=image,
            exposure_ms=exposure_ms,
            binning=binning,
            add_poisson=add_poisson,
            xp=xp,
        )
        if binning > 1:
            yx = dict.fromkeys(image.dims[-2:], binning)
            image = image.coarsen(yx, boundary="trim").mean()
        return DataArray(
            new_data, dims=image.dims, coords=image.coords, attrs=image.attrs
        )
