from pydantic_settings import BaseSettings

from .backend import BackendName, DeviceName, NumpyAPI


class Settings(BaseSettings):
    np_backend: BackendName = "auto"
    device: DeviceName = "auto"

    def backend_module(self) -> NumpyAPI:
        return NumpyAPI.create(self.np_backend)
