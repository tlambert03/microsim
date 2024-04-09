from typing import Any

from pydantic import BaseModel, model_validator

from microsim._data_array import DataArray
from microsim.schema.backend import NumpyAPI
from microsim.schema.spectrum import Spectrum

from .matslines import MatsLines

Distribution = MatsLines


class Fluorophore(BaseModel):
    name: str
    excitation_spectrum: Spectrum
    emission_spectrum: Spectrum
    bleaching_half_life_s: float | None = None
    lifetime_ns: float | None = None


class FluorophoreDistribution(BaseModel):
    distribution: Distribution
    fluorophore: Fluorophore | None = None

    def render(self, space: DataArray, xp: NumpyAPI | None = None) -> DataArray:
        return self.distribution.render(space, xp)

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, Distribution):
            return {"distribution": value}
        if isinstance(value, dict):
            if "distribution" not in value and "type" in value:
                return {"distribution": value}
        return value


class Sample(BaseModel):
    labels: list[FluorophoreDistribution]
