from typing import Literal

from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum


class _Filter(SimBaseModel):
    type: str
    name: str = ""


class Bandpass(_Filter):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: float
    bandwidth: float
    transmission: float = 1.0


class Shortpass(_Filter):
    type: Literal["shortpass"] = "shortpass"
    cutoff: float
    slope: float = 1.0
    transmission: float = 1.0

    @property
    def bandcenter(self) -> float:
        return self.cutoff


class Longpass(_Filter):
    type: Literal["longpass"] = "longpass"
    cutoff: float
    slope: float = 1.0
    transmission: float = 1.0

    @property
    def bandcenter(self) -> float:
        return self.cutoff


class FilterSpectrum(_Filter):
    type: Literal["spectrum"] = "spectrum"
    spectrum: Spectrum

    @property
    def bandcenter(self) -> float:
        return self.spectrum.peak_wavelength


Filter = Bandpass | Shortpass | Longpass | FilterSpectrum
