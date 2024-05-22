from typing import Annotated, Literal, Protocol

from annotated_types import Interval

from microsim._field_types import Nanometers
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

Transmission = Annotated[float, Interval(ge=0, le=1.0)]


class Filter(Protocol):
    type: str
    name: str = ""

    @property
    def bandcenter(self) -> Nanometers: ...

    @property
    def spectrum(self) -> Spectrum: ...


class _FilterBase(SimBaseModel):
    type: str
    name: str = ""


class Bandpass(_FilterBase):
    type: Literal["bandpass"] = "bandpass"
    bandcenter: Nanometers
    bandwidth: Nanometers
    transmission: Transmission = 1.0

    @property
    def spectrum(self) -> Spectrum:
        raise NotImplementedError()


class Shortpass(_FilterBase):
    type: Literal["shortpass"] = "shortpass"
    cutoff: Nanometers
    slope: float = 1.0
    transmission: Transmission = 1.0

    @property
    def bandcenter(self) -> Nanometers:
        return self.cutoff

    @property
    def spectrum(self) -> Spectrum:
        raise NotImplementedError()


class Longpass(_FilterBase):
    type: Literal["longpass"] = "longpass"
    cutoff: Nanometers
    slope: float = 1.0
    transmission: Transmission = 1.0

    @property
    def bandcenter(self) -> Nanometers:
        return self.cutoff

    @property
    def spectrum(self) -> Spectrum:
        raise NotImplementedError()


class FilterSpectrum(_FilterBase):
    type: Literal["spectrum"] = "spectrum"
    spectrum: Spectrum

    @property
    def bandcenter(self) -> Nanometers:
        return self.spectrum.peak_wavelength


class FilterPlacement(SimBaseModel):
    # where EX = excitation, EM = emission, BS = beam splitter, BSi = inverted BS
    spectrum: Spectrum
    path: Literal["EX", "EM", "BS", "BSi"]
    name: str = ""
    type: str = ""

    @property
    def reflects_emission(self) -> bool:
        return self.path == "BSi"

    @property
    def bandcenter(self) -> Nanometers: ...
