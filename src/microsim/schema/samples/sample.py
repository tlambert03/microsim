from typing import Any

import xarray as xr
from pydantic import BaseModel, model_validator

from microsim.schema.backend import NumpyAPI

from .matslines import MatsLines

Distribution = MatsLines


class Spectrum(BaseModel):
    wavelength: list[float]  # nm
    intensity: list[float]  # normalized to 1
    scalar: float = 1  # scalar to multiply intensity by, such as EC or QY


class Fluorophore(BaseModel):
    name: str
    excitation_spectrum: Spectrum
    emission_spectrum: Spectrum
    bleaching_half_life_s: float | None = None
    lifetime_ns: float | None = None


class FluorophoreDistribution(BaseModel):
    distribution: Distribution
    fluorophore: Fluorophore | None = None

    def render(self, space: xr.DataArray, xp: NumpyAPI | None = None) -> xr.DataArray:
        return self.distribution.render(space, xp)

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, Distribution):
            return {"distribution": value}
        return value


class Sample(BaseModel):
    labels: list[FluorophoreDistribution]
