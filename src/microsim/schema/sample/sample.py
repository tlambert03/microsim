from typing import Any, get_args

from pydantic import Field, model_validator

from microsim._data_array import DataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI

from .cosem import Cosem
from .fluorophore import Fluorophore
from .matslines import MatsLines

Distribution = MatsLines | Cosem


class FluorophoreDistribution(SimBaseModel):
    distribution: Distribution = Field(..., discriminator="type")
    fluorophore: Fluorophore | None = None

    def cache_path(self) -> tuple[str, ...] | None:
        if not hasattr(self.distribution, "cache_path"):
            return None
        return self.distribution.cache_path()

    def render(self, space: DataArray, xp: NumpyAPI | None = None) -> DataArray:
        return self.distribution.render(space, xp)

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, get_args(Distribution)):
            return {"distribution": value}
        if isinstance(value, dict):
            if "distribution" not in value and "type" in value:
                return {"distribution": value}
        return value


class Sample(SimBaseModel):
    labels: list[FluorophoreDistribution]
