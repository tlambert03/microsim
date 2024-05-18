from typing import Literal

from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum
import numpy as np

class _Filter(SimBaseModel):
    type: str
    name: str = ""


class Bandpass(_Filter):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: float
    bandwidth: float
    transmission: float = 1.0

    def get_spectrum(self) -> Spectrum:
        start = self.bandcenter - self.bandwidth / 2
        end = self.bandcenter + self.bandwidth / 2
        wavelength = list(range(start, end+1, 1))
        intensity = [1.0] * len(wavelength)
        return Spectrum(wavelength.tolist(), intensity.tolist())

class Shortpass(_Filter):
    type: Literal["shortpass"] = "shortpass"
    cutoff: float
    slope: float = 1.0
    transmission: float = 1.0

    @property
    def bandcenter(self) -> float:
        return self.cutoff

    def get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")

class Longpass(_Filter):
    type: Literal["longpass"] = "longpass"
    cutoff: float
    slope: float = 1.0
    transmission: float = 1.0

    @property
    def bandcenter(self) -> float:
        return self.cutoff

    def get_spectrum(self) -> Spectrum:
        raise NotImplementedError("Needs to be implemented")

class FilterSpectrum(_Filter):
    type: Literal["spectrum"] = "spectrum"
    spectrum: Spectrum

    @property
    def bandcenter(self) -> float:
        return self.spectrum.peak_wavelength

    def get_spectrum(self) -> Spectrum:
        return self.spectrum

Filter = Bandpass | Shortpass | Longpass | FilterSpectrum
