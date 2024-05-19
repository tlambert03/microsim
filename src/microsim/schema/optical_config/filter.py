from functools import cached_property
from typing import Literal

import numpy as np
from pydantic import computed_field

from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum


class _Filter(SimBaseModel):
    type: str
    name: str = ""

    @computed_field
    @cached_property
    def spectrum(self) -> Spectrum:
        return self._get_spectrum()

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")


class Bandpass(_Filter):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: float
    bandwidth: float
    transmission: float = 1.0

    def _get_spectrum(self) -> Spectrum:
        start = self.bandcenter - self.bandwidth / 2
        end = self.bandcenter + self.bandwidth / 2
        wavelength = np.arange(start, end + 1, 1)
        intensity = np.ones_like(wavelength)
        return Spectrum(wavelength=wavelength, intensity=intensity)


class Shortpass(_Filter):
    type: Literal["shortpass"] = "shortpass"
    cutoff: float
    slope: float = 1.0
    transmission: float = 1.0

    @property
    def bandcenter(self) -> float:
        return self.cutoff

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")


class Longpass(_Filter):
    type: Literal["longpass"] = "longpass"
    cutoff: float
    slope: float = 1.0
    transmission: float = 1.0

    @property
    def bandcenter(self) -> float:
        return self.cutoff

    def _get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")


class FilterSpectrum(_Filter):
    type: Literal["spectrum"] = "spectrum"
    spectrum_data: Spectrum

    @property
    def bandcenter(self) -> float:
        return self.spectrum.peak_wavelength

    def _get_spectrum(self) -> Spectrum:
        return self.spectrum_data


Filter = Bandpass | Shortpass | Longpass | FilterSpectrum
