from functools import cached_property
from typing import Annotated, Literal

import numpy as np
from annotated_types import Interval
from pydantic import computed_field

from microsim._field_types import Nanometers
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

Transmission = Annotated[float, Interval(ge=0, le=1.0)]


class _Filter(SimBaseModel):
    type: str
    name: str = ""

    @computed_field  # type: ignore
    @cached_property
    def spectrum(self) -> Spectrum:
        return self._get_spectrum()

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")


class Bandpass(_Filter):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: Nanometers
    bandwidth: Nanometers
    transmission: Transmission = 1.0

    def _get_spectrum(self) -> Spectrum:
        start = self.bandcenter - self.bandwidth / 2
        end = self.bandcenter + self.bandwidth / 2
        wavelength = np.arange(start, end + 1, 1)
        intensity = np.ones_like(wavelength)
        return Spectrum(wavelength=wavelength, intensity=intensity)


class Shortpass(_Filter):
    type: Literal["shortpass"] = "shortpass"
    cutoff: Nanometers
    slope: float = 1.0
    transmission: Transmission = 1.0

    @property
    def bandcenter(self) -> Nanometers:
        return self.cutoff

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")


class Longpass(_Filter):
    type: Literal["longpass"] = "longpass"
    cutoff: Nanometers
    slope: float = 1.0
    transmission: Transmission = 1.0

    @property
    def bandcenter(self) -> Nanometers:
        return self.cutoff

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")


class FilterSpectrum(_Filter):
    type: Literal["spectrum"] = "spectrum"
    spectrum_data: Spectrum

    @property
    def bandcenter(self) -> Nanometers:
        return self.spectrum.peak_wavelength

    def _get_spectrum(self) -> Spectrum:
        return self.spectrum_data


Filter = Bandpass | Shortpass | Longpass | FilterSpectrum
