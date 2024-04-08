from typing import Literal

from pydantic_settings import BaseSettings

from .backend import NumpyAPI

BackendName = Literal["numpy", "torch", "jax", "cupy", "auto"]
DeviceName = Literal["cpu", "gpu", "auto"]


class Settings(BaseSettings):
    np_backend: BackendName = "auto"
    device: DeviceName = "auto"

    def backend_module(self) -> NumpyAPI:
        return NumpyAPI.create(self.np_backend)
