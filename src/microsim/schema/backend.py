import warnings
from contextlib import nullcontext, suppress
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import jax

DeviceName = Literal["cpu", "gpu", "auto"]
BackendName = Literal["numpy", "torch", "jax", "cupy", "auto"]
NumpyAPIType = TypeVar("NumpyAPIType", bound="NumpyAPI")
ArrT = TypeVar("ArrT", bound=npt.ArrayLike)


class NumpyAPI:
    @classmethod
    def create(cls, backend: "BackendName | NumpyAPI | None") -> "NumpyAPI":
        if isinstance(backend, cls):
            return backend
        if not backend:
            backend = backend or "auto"

        ctx = suppress(ImportError) if backend == "auto" else nullcontext()
        if backend in ("cupy", "auto"):
            with ctx:
                return CupyAPI()
        elif backend in ("jax", "auto"):
            with ctx:
                return JaxAPI()
        elif backend in ("torch", "auto"):
            with ctx:
                return TorchAPI()

        return NumpyAPI()

    def __init__(self) -> None:
        from scipy import signal, special

        self.xp = np
        self.signal = signal
        self.j0 = special.j0
        self.j1 = special.j1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.xp, name)

    # def zeros(
    #     self, shape: Sequence[int], dtype: npt.DTypeLike = np.float32
    # ) -> npt.ArrayLike:
    #     return self.xp.zeros(shape, dtype)

    def fftconvolve(
        self, a: ArrT, b: ArrT, mode: Literal["full", "valid", "same"] = "full"
    ) -> ArrT:
        return self.signal.fftconvolve(a, b, mode=mode)  # type: ignore

    def map_coordinates(
        self, input: npt.ArrayLike, coordinates: npt.ArrayLike, order: int = 3
    ) -> npt.NDArray:
        from scipy.ndimage import map_coordinates

        return map_coordinates(input, coordinates, order=order)  # type: ignore

    def _simp_like(self, arr: npt.ArrayLike) -> npt.ArrayLike:
        simp = self.xp.empty_like(arr)
        simp[::2] = 4
        simp[1::2] = 2
        simp[-1] = 1
        return simp

    def _array_assign(
        self, arr: npt.ArrayLike, mask: npt.ArrayLike, value: npt.ArrayLike
    ) -> npt.ArrayLike:
        arr[mask] = value  # type: ignore
        return arr


class JaxAPI(NumpyAPI):
    def __init__(self) -> None:
        import jax
        from jax.scipy import signal
        from jax.scipy.ndimage import map_coordinates

        from ._jax_bessel import j0, j1

        self.xp = jax.numpy
        self.signal = signal
        self.map_coordinates = map_coordinates  # type: ignore
        self.j0 = j0
        self.j1 = j1

    @property
    def random(self) -> ModuleType:
        return np.random

    def fftconvolve(
        self, a: ArrT, b: ArrT, mode: Literal["full", "valid", "same"] = "full"
    ) -> ArrT:
        return self.signal.fftconvolve(a, b, mode=mode)  # type: ignore

    def _simp_like(self, arr: "jax.Array") -> "jax.Array":  # type: ignore
        simp = self.xp.empty_like(arr)

        simp = simp.at[::2].set(4)
        simp = simp.at[1::2].set(2)
        simp = simp.at[-1].set(1)
        return simp

    def _array_assign(  # type: ignore
        self,
        arr: "jax.Array",
        mask: npt.ArrayLike,
        value: "jax.Array",
    ) -> "jax.Array":
        return arr.at[mask].set(value)


class CupyAPI(NumpyAPI):
    def __init__(self) -> None:
        import cupy
        from cupyx.scipy.ndimage import map_coordinates

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from cupyx.scipy import signal

        self.xp = cupy
        self.signal = signal
        self.map_coordinates = map_coordinates  # type: ignore

    def fftconvolve(
        self, a: ArrT, b: ArrT, mode: Literal["full", "valid", "same"] = "full"
    ) -> ArrT:
        return self.signal.fftconvolve(a, b, mode=mode)  # type: ignore


class TorchAPI(NumpyAPI):
    def __init__(self) -> None:
        import torch

        self.xp = torch

    def fftconvolve(
        self, a: ArrT, b: ArrT, mode: Literal["full", "valid", "same"] = "full"
    ) -> ArrT:
        raise NotImplementedError("fftconvolve not implemented for torch")
