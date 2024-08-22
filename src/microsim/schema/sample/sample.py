from collections.abc import Callable
from typing import Any, get_args

import numpy as np
from pydantic import Field, model_validator

from microsim._data_array import ArrayProtocol, xrDataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI

from ._distributions.cosem import CosemLabel
from ._distributions.direct import FixedArrayTruth
from ._distributions.matslines import MatsLines
from .fluorophore import Fluorophore, Spectrum

Distribution = MatsLines | CosemLabel | FixedArrayTruth
DistributionTypes = get_args(Distribution)


waves = np.arange(300, 800, 1)
MOCK_FLUOR = Fluorophore(
    name="mockFluorophore",
    excitation_spectrum=Spectrum(wavelength=waves, intensity=np.ones_like(waves)),
    emission_spectrum=Spectrum(wavelength=waves, intensity=np.ones_like(waves)),
    extinction_coefficient=50_000,
    quantum_yield=1,
    lifetime_ns=1,
)


class FluorophoreDistribution(SimBaseModel):
    distribution: Distribution = Field(...)
    fluorophore: Fluorophore = MOCK_FLUOR
    # either a scalar that will be multiplied by the distribution
    # (e.g. to increase/decrease concentration of fluorophore)
    # or a function that will be applied to the distribution
    # (e.g. to add noise, labeling randomness/inefficiency, etc...)
    density_scaler: float | Callable[[xrDataArray], xrDataArray] | None = None

    def __hash__(self) -> int:
        return id(self)

    def cache_path(self) -> tuple[str, ...] | None:
        if hasattr(self.distribution, "cache_path"):
            return self.distribution.cache_path()
        return None

    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        dist = self.distribution.render(space, xp)
        if isinstance(self.density_scaler, float):
            return dist * self.density_scaler
        elif callable(self.density_scaler):
            return self.density_scaler(dist)
        return dist

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

    def __str__(self) -> str:
        return f"{self.fluorophore.name} - {self.distribution.__class__.__name__}"


class Sample(SimBaseModel):
    labels: list[FluorophoreDistribution]

    @model_validator(mode="before")
    def _validate_labels(cls, value: Any) -> Any:
        if isinstance(value, list | tuple):
            return {"labels": list(value)}
        return value
