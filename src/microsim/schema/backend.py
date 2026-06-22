"""The single JAX array backend.

microsim is JAX-only: the forward model is built from `jax.numpy` so that it is
differentiable end-to-end.  This module exposes one small backend object (still
named `NumpyAPI` for historical reasons) that wraps `jax.numpy`, manages a
PRNG key for the stochastic detector/sample steps, and provides a couple of
helpers (`poisson_rvs`, `norm_rvs`, `fftconvolve`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from microsim._data_array import ArrayProtocol

# kept for backwards-compatible signatures; only "jax" is meaningful now
BackendName = Literal["jax", "auto"]
DeviceName = Literal["cpu", "gpu", "auto"]


class NumpyAPI:
    """JAX array backend (single backend; `jax.numpy` under the hood)."""

    _random_seed: int | None = None
    _float_dtype: np.dtype | None = None

    def __init__(self) -> None:
        self.xp = jnp
        self.signal = jax.scipy.signal
        self.stats = jax.scipy.stats
        self._key = jax.random.PRNGKey(0)

    @classmethod
    def create(cls, backend: BackendName | NumpyAPI | None = None) -> NumpyAPI:
        """Return the JAX backend (the `backend` argument is ignored)."""
        if isinstance(backend, NumpyAPI):
            return backend
        return cls()

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
        self._key = jax.random.PRNGKey(seed)
        np.random.seed(seed)  # host-side sample generators (e.g. MatsLines)

    def _next_key(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub

    def __getattr__(self, name: str) -> Any:
        return getattr(self.xp, name)

    def asarray(
        self, x: npt.ArrayLike, dtype: npt.DTypeLike | None = None
    ) -> jax.Array:
        return jnp.asarray(x, dtype=dtype)

    def zeros(
        self, shape: int | Sequence[int], dtype: npt.DTypeLike = None
    ) -> jax.Array:
        return jnp.zeros(shape, dtype=dtype or self.float_dtype)

    def ones(
        self, shape: int | Sequence[int], dtype: npt.DTypeLike = None
    ) -> jax.Array:
        return jnp.ones(shape, dtype=dtype or self.float_dtype)

    def poisson_rvs(
        self, lam: npt.ArrayLike, shape: Sequence[int] | None = None
    ) -> jax.Array:
        return jax.random.poisson(self._next_key(), jnp.asarray(lam), shape=shape)

    def norm_rvs(
        self, loc: ArrayProtocol, scale: npt.ArrayLike | None = None
    ) -> jax.Array:
        loc = jnp.asarray(loc)
        std = jax.random.normal(self._next_key(), shape=loc.shape)
        return std * (1.0 if scale is None else scale) + loc

    def fftconvolve(
        self, a: ArrayProtocol, b: ArrayProtocol, mode: str = "full"
    ) -> jax.Array:
        return self.signal.fftconvolve(jnp.asarray(a), jnp.asarray(b), mode=mode)

    def _array_assign(self, arr: jax.Array, mask: Any, value: Any) -> jax.Array:
        return jnp.asarray(arr).at[mask].set(value)

    def __hash__(self) -> int:
        return hash((type(self), self._random_seed))

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other)
