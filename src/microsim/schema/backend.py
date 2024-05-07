from __future__ import annotations

import warnings
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType
    from typing import SupportsIndex, TypeAlias

    import jax

    _Shape: TypeAlias = tuple[int, ...]

    # Anything that can be coerced to a shape tuple
    _ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]

DeviceName = Literal["cpu", "gpu", "auto"]
BackendName = Literal["numpy", "torch", "jax", "cupy", "auto"]
NumpyAPIType = TypeVar("NumpyAPIType", bound="NumpyAPI")
ArrT = TypeVar("ArrT", bound=npt.ArrayLike)


class NumpyAPI:
    @classmethod
    def create(cls, backend: BackendName | NumpyAPI | None) -> NumpyAPI:
        if isinstance(backend, NumpyAPI):
            return backend
        if not backend:
            backend = backend or "auto"
        backend = backend.lower()  # type: ignore

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
        from scipy import signal, special, stats
        from scipy.ndimage import map_coordinates

        self._random_seed: int | None = None
        self.xp = np
        self.signal = signal
        self.stats = stats
        self.j0 = special.j0
        self.j1 = special.j1
        self.map_coordinates = map_coordinates
        self._float_dtype: np.dtype | None = None

    @property
    def float_dtype(self) -> np.dtype | None:
        return self._float_dtype

    @float_dtype.setter
    def float_dtype(self, dtype: npt.DTypeLike) -> None:
        self._float_dtype = np.dtype(dtype)
        if not np.issubdtype(self._float_dtype, np.floating):
            raise ValueError(
                f"Expected a floating-point dtype, got {self._float_dtype}"
            )

    def set_random_seed(self, seed: int) -> None:
        self._random_seed = seed
        self.xp.random.seed(seed)

    def asarray(
        self, x: npt.ArrayLike, dtype: npt.DTypeLike | None = None
    ) -> npt.NDArray:
        return self.xp.asarray(x, dtype=dtype)

    def zeros(
        self, shape: int | Sequence[int], dtype: npt.DTypeLike = None
    ) -> npt.NDArray:
        if dtype is None:
            dtype = self.float_dtype
        return self.xp.zeros(shape, dtype=dtype)

    def ones(
        self, shape: int | Sequence[int], dtype: npt.DTypeLike = None
    ) -> npt.NDArray:
        if dtype is None:
            dtype = self.float_dtype
        return self.xp.ones(shape, dtype=dtype)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.xp, name)

    def poisson_rvs(
        self, lam: npt.ArrayLike, shape: Sequence[int] | None = None
    ) -> npt.NDArray:
        return self.stats.poisson.rvs(lam, size=shape)  # type: ignore

    def fftconvolve(
        self, a: ArrT, b: ArrT, mode: Literal["full", "valid", "same"] = "full"
    ) -> ArrT:
        return self.signal.fftconvolve(a, b, mode=mode)  # type: ignore

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

    # WARNING: these hash and eq methods may be problematic later?
    # the goal is to make any instance of a NumpyAPI hashable and equal to any
    # other instance, as long as they are of the same type and random seed.
    def __hash__(self) -> int:
        return hash(type(self)) + hash(self._random_seed)

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other)


class JaxAPI(NumpyAPI):
    def __init__(self) -> None:
        import jax
        from jax.random import PRNGKey
        from jax.scipy import signal, stats
        from jax.scipy.ndimage import map_coordinates

        from ._jax_bessel import j0, j1

        self._random_seed: int | None = None
        self.xp = jax.numpy
        self.signal = signal
        self.stats = stats
        self.map_coordinates = map_coordinates
        self.j0 = j0
        self.j1 = j1
        self._key = PRNGKey(0)

    @property
    def random(self) -> ModuleType:  # TODO
        return np.random

    def set_random_seed(self, seed: int) -> None:
        from jax.random import PRNGKey

        self._random_seed = seed
        self._key = PRNGKey(seed)
        # FIXME
        # tricky... we actually still do use the numpy random seed in addition to
        # the jax key.  It would be nice to get rid of this line while keeping the
        # tests passing.
        np.random.seed(seed)

    def poisson_rvs(  # type: ignore
        self,
        lam: jax.Array | float,
        shape: Sequence[int] | None = None,
    ) -> jax.Array:
        from jax.random import poisson

        return poisson(self._key, lam, shape=shape)

    def fftconvolve(
        self, a: ArrT, b: ArrT, mode: Literal["full", "valid", "same"] = "full"
    ) -> ArrT:
        return self.signal.fftconvolve(a, b, mode=mode)  # type: ignore[no-any-return]

    def _simp_like(self, arr: jax.Array) -> jax.Array:
        simp = self.xp.empty_like(arr)

        simp = simp.at[::2].set(4)
        simp = simp.at[1::2].set(2)
        simp = simp.at[-1].set(1)
        return simp

    def _array_assign(  # type: ignore
        self,
        arr: jax.Array,
        mask: npt.ArrayLike,
        value: jax.Array,
    ) -> jax.Array:
        return arr.at[mask].set(value)


class CupyAPI(NumpyAPI):
    def __init__(self) -> None:
        import cupy
        from cupyx.scipy.ndimage import map_coordinates

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from cupyx.scipy import signal, special, stats

        self.xp = cupy
        self.signal = signal
        self.stats = stats
        self.j0 = special.j0
        self.j1 = special.j1
        self.map_coordinates = map_coordinates

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
