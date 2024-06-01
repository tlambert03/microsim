import random
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from microsim._field_types import FloatDtype

from ._base_model import SimBaseModel
from .backend import BackendName, DeviceName, NumpyAPI


class Settings(SimBaseModel, BaseSettings):
    np_backend: BackendName = "auto"
    device: DeviceName = "auto"
    float_dtype: FloatDtype = Field(  # type: ignore[assignment]
        "float32",
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

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        validate_assignment=True
    )

    def backend_module(self) -> NumpyAPI:
        backend = NumpyAPI.create(self.np_backend)
        backend.float_dtype = self.float_dtype
        if self.random_seed is not None:
            backend.set_random_seed(self.random_seed)
        return backend
