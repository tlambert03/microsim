from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from pydantic import Field, computed_field

from microsim import fpbase
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

if TYPE_CHECKING:
    from typing import Self

Transmission = Annotated[float, Interval(ge=0, le=1.0)]


class Placement(Enum):
    EX_PATH = "EX"
    EM_PATH = "EM"
    BS = "BS"
    BS_INV = "BSi"  # inverted beam splitter that reflects emission & transmits ex
    ALL = "ALL"  # implies that the filter should be used regardless of placement


class _FilterBase(SimBaseModel):
    type: str = ""
    name: str = ""
    placement: Placement = Placement.ALL

    @computed_field  # type: ignore
    @cached_property
    def spectrum(self) -> Spectrum:
        return self._get_spectrum()

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError()

    def inverted(self) -> "Self":
        return self.model_copy(update={"spectrum": self.spectrum.inverted()})

    def center_wave(self) -> float:
        """Return the weighted mean wavelength of the filter."""
        avg = np.average(self.spectrum.wavelength, weights=self.spectrum.intensity)
        return float(avg)

    @classmethod
    def from_fpbase(
        cls,
        filter: str | fpbase.FilterPlacement | fpbase.FilterSpectrum,
        placement: Placement = Placement.ALL,
    ) -> "SpectrumFilter":
        if isinstance(filter, str):
            filter = fpbase.get_filter(filter)  # noqa
        elif isinstance(filter, fpbase.FilterPlacement):
            return SpectrumFilter(
                name=filter.name,
                placement=filter.path,
                transmission=Spectrum.from_fpbase(filter.spectrum),
            )
        if not isinstance(filter, fpbase.FilterSpectrum):
            raise TypeError(
                "filter must be a string, FilterPlacement, or FilterSpectrum, "
                f"not {type(filter)}"
            )

        return SpectrumFilter(
            name=filter.ownerFilter.name,
            placement=placement,
            transmission=Spectrum.from_fpbase(filter),
        )

    def plot(self) -> None:
        self.spectrum.plot()  # type: ignore [call-arg]


class Bandpass(_FilterBase):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: float  # nm
    bandwidth: float  # nm
    transmission: Transmission = 1.0

    def center_wave(self) -> float:
        return self.bandcenter

    def _get_spectrum(self) -> Spectrum:
        min_wave = min(300, (self.bandcenter - self.bandwidth))
        max_wave = max(800, (self.bandcenter + self.bandwidth))
        wavelength = np.arange(min_wave, max_wave, 1)
        return Spectrum(
            wavelength=wavelength,
            intensity=bandpass(
                wavelength,
                center=self.bandcenter,
                bandwidth=self.bandwidth,
                transmission=self.transmission,
            ),
        )


class Shortpass(_FilterBase):
    type: Literal["shortpass"] = "shortpass"
    cutoff: float  # nm
    slope: float | None = None
    transmission: Transmission = 1.0
    placement: Placement = Placement.EX_PATH

    def center_wave(self) -> float:
        raise NotImplementedError("center wave is not defined for shortpass filters")

    def _get_spectrum(self) -> Spectrum:
        min_wave = min(300, self.cutoff - 50)
        max_wave = max(800, self.cutoff + 50)
        wavelength = np.arange(min_wave, max_wave, 1)
        return Spectrum(
            wavelength=wavelength,
            intensity=sigmoid(
                wavelength,
                self.cutoff,
                slope=self.slope or 5,
                up=False,
                max=self.transmission,
            ),
        )


class Longpass(_FilterBase):
    type: Literal["longpass"] = "longpass"
    cuton: float  # nm
    slope: float | None = None
    transmission: Transmission = 1.0
    placement: Placement = Placement.EM_PATH

    def center_wave(self) -> float:
        raise NotImplementedError("center wave is not defined for longpass filters")

    def _get_spectrum(self) -> Spectrum:
        min_wave = min(300, self.cuton - 50)
        max_wave = max(800, self.cuton + 50)
        wavelength = np.arange(min_wave, max_wave, 1)
        return Spectrum(
            wavelength=wavelength,
            intensity=sigmoid(
                wavelength,
                self.cuton,
                slope=self.slope or 5,
                up=True,
                max=self.transmission,
            ),
        )


class SpectrumFilter(_FilterBase):
    type: Literal["spectrum"] = "spectrum"
    transmission: Spectrum = Field(..., repr=False)  # because of spectrum on super()

    def _get_spectrum(self) -> Spectrum:
        return self.transmission


Filter = Bandpass | Shortpass | Longpass | SpectrumFilter


def sigmoid(
    wavelength: Any, cutoff: float, slope: float = 1, max: float = 1, up: bool = True
) -> Any:
    if up:
        slope = -slope
    with np.errstate(over="ignore"):
        return max / (1 + np.exp(slope * (wavelength - cutoff)))


def bandpass(
    wavelength: Any,
    center: float | Sequence[float],
    bandwidth: float | Sequence[float],
    slope: float = 5,
    transmission: float = 1,
) -> Any:
    if isinstance(center, Sequence):
        if isinstance(bandwidth, Sequence):
            if len(center) != len(bandwidth):
                raise ValueError("center and bandwidth must have the same length")
        else:
            bandwidth = [bandwidth] * len(center)

        segments = [
            bandpass(wavelength, c, b, slope=slope, transmission=transmission)
            for c, b in zip(center, bandwidth, strict=False)
        ]
        return np.prod(segments, axis=0)
    elif isinstance(bandwidth, Sequence):
        raise ValueError("center and bandwidth must have the same shape")

    left = sigmoid(wavelength, center - bandwidth / 2, slope=slope)
    right = sigmoid(wavelength, center + bandwidth / 2, slope=slope, up=False)
    return left * right * transmission
