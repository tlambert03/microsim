import random

from pydantic import Field
from pydantic_settings import BaseSettings

from .backend import BackendName, DeviceName, NumpyAPI


class Settings(BaseSettings):
    np_backend: BackendName = "auto"
    device: DeviceName = "auto"
    random_seed: int | None = Field(
        default_factory=lambda: random.randint(0, 2**32 - 1)
    )

    def backend_module(self) -> NumpyAPI:
        backend = NumpyAPI.create(self.np_backend)
        if self.random_seed is not None:
            backend.random.seed(self.random_seed)
        return backend
