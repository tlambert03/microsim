import random
from typing import Annotated, Any, ClassVar

import numpy as np
from pydantic import Field
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import PlainValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ._base_model import SimBaseModel
from .backend import BackendName, DeviceName, NumpyAPI


def _np_float_dtype(x: Any) -> np.dtype:
    dt = np.dtype(x)
    if not np.issubdtype(dt, np.floating):
        raise ValueError(f"Expected a floating-point dtype, got {dt}")
    return dt


NpDtype = Annotated[
    np.dtype,
    PlainValidator(_np_float_dtype),
    PlainSerializer(lambda x: str(x), return_type=str),
]


class Settings(SimBaseModel, BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        validate_assignment=True
    )

    np_backend: BackendName = "auto"
    device: DeviceName = "auto"
    float_dtype: NpDtype = Field(
        np.dtype("float32"),
        description="Floating-point precision to use for simulations.",
    )
    random_seed: int | None = Field(
        default_factory=lambda: random.randint(0, 2**32 - 1)
    )
    max_psf_radius_aus: float | None = Field(
        8,
        description=(
            "When simulating, restrict generated lateral PSF size to no more than this "
            "many Airy units. Decreasing this can *dramatically* speed up simulations, "
            "but will decrease accuracy. If `None`, no restriction is applied, and the "
            "psf will be generated to the full extent of the simulated space."
        ),
    )

    def backend_module(self) -> NumpyAPI:
        backend = NumpyAPI.create(self.np_backend)
        backend.float_dtype = self.float_dtype
        if self.random_seed is not None:
            backend.set_random_seed(self.random_seed)
        return backend
