from typing import Annotated, Literal

from annotated_types import Interval

from microsim._field_types import Nanometers
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

Transmission = Annotated[float, Interval(ge=0, le=1.0)]


class _Filter(SimBaseModel):
    type: str
    name: str = ""


class Bandpass(_Filter):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: Nanometers
    bandwidth: Nanometers
    transmission: Transmission = 1.0


class Shortpass(_Filter):
    type: Literal["shortpass"] = "shortpass"
    cutoff: Nanometers
    slope: float = 1.0
    transmission: Transmission = 1.0

    @property
    def bandcenter(self) -> Nanometers:
        return self.cutoff


class Longpass(_Filter):
    type: Literal["longpass"] = "longpass"
    cutoff: Nanometers
    slope: float = 1.0
    transmission: Transmission = 1.0

    @property
    def bandcenter(self) -> Nanometers:
        return self.cutoff


class FilterSpectrum(_Filter):
    type: Literal["spectrum"] = "spectrum"
    spectrum: Spectrum

    @property
    def bandcenter(self) -> Nanometers:
        return self.spectrum.peak_wavelength


Filter = Bandpass | Shortpass | Longpass | FilterSpectrum
