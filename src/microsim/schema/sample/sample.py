from typing import Any, get_args

from pydantic import Field, model_validator

from microsim._data_array import ArrayProtocol, xrDataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI

from ._distributions.cosem import CosemLabel
from ._distributions.direct import FixedArrayTruth
from ._distributions.matslines import MatsLines
from .fluorophore import Fluorophore

Distribution = MatsLines | CosemLabel | FixedArrayTruth
DistributionTypes = get_args(Distribution)


class FluorophoreDistribution(SimBaseModel):
    distribution: Distribution = Field(...)
    fluorophore: Fluorophore | None = None

    def cache_path(self) -> tuple[str, ...] | None:
        if hasattr(self.distribution, "cache_path"):
            return self.distribution.cache_path()
        return None

    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        return self.distribution.render(space, xp)

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, DistributionTypes):
            return {"distribution": value}
        if isinstance(value, dict):
            if "distribution" not in value and "type" in value:
                return {"distribution": value}
        return value

    @classmethod
    def from_array(cls, array: ArrayProtocol) -> "FluorophoreDistribution":
        return cls(distribution=FixedArrayTruth(array=array))


class Sample(SimBaseModel):
    labels: list[FluorophoreDistribution]
