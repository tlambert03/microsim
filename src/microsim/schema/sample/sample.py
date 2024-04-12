from typing import Any

from pydantic import BaseModel, Field, model_validator

from microsim._data_array import DataArray
from microsim.schema.backend import NumpyAPI

from .fluorophore import Fluorophore
from .matslines import MatsLines

Distribution = MatsLines


class FluorophoreDistribution(BaseModel):
    distribution: Distribution = Field(..., discriminator="type")
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