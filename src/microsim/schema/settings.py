from types import ModuleType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic_settings import BaseSettings


class NumpyAPI:
    def __init__(self) -> None:
        self.xp = np

    def zeros(self, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float32):
        return self.xp.zeros(shape, dtype)

    def fftconvolve(
        self,
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        mode: Literal["full", "valid", "same"] = "full",
    ):
        from scipy import signal

        return signal.fftconvolve(a, b, mode=mode)

    def __getattr__(self, name) -> Any:
        return getattr(self.xp, name)


class JaxAPI(NumpyAPI):
    def __init__(self) -> None:
        import jax

        self.xp = jax.numpy

    @property
    def random(self) -> ModuleType:
        return np.random

    def fftconvolve(
        self,
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        mode: Literal["full", "valid", "same"] = "full",
    ):
        from jax.scipy import signal

        return signal.fftconvolve(a, b, mode=mode)


class TorchAPI(NumpyAPI):
    def __init__(self) -> None:
        import torch

        self.xp = torch

    def fftconvolve(
        self,
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        mode: Literal["full", "valid", "same"] = "full",
    ):
        raise NotImplementedError("fftconvolve not implemented for torch")


class CupyAPI(NumpyAPI):
    def __init__(self) -> None:
        import cupy

        self.xp = cupy

    def fftconvolve(
        self,
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        mode: Literal["full", "valid", "same"] = "full",
    ):
        # silence warnings
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from cupyx.scipy import signal

        return signal.fftconvolve(a, b, mode=mode)


class Settings(BaseSettings):
    np_backend: Literal["numpy", "torch", "jax", "cupy", "auto"] = "auto"
    device: Literal["cpu", "gpu", "auto"] = "auto"

    def backend_module(self) -> NumpyAPI:
        if self.np_backend in ("cupy", "auto"):
            return CupyAPI()
        elif self.np_backend in ("jax", "auto"):
            return JaxAPI()
        elif self.np_backend in ("torch", "auto"):
            return TorchAPI()
        else:
            return NumpyAPI()
