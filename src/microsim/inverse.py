"""Small helpers for gradient-based inverse problems on the forward model.

microsim's forward model (PSF generation, convolution, detection) is written in
JAX and is therefore differentiable.  This module provides a few reusable pieces
for phrasing inverse problems - phase retrieval, deconvolution, PSF engineering
- as `optax` optimizations.  See `examples/inverse/` for complete scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PyTree = object


def normalized_mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Scale-invariant MSE: `mean((pred-target)²) / mean(target²)`."""
    return jnp.mean((pred - target) ** 2) / jnp.mean(target**2)


def poisson_nll(rate: jax.Array, counts: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Poisson negative log-likelihood (up to a constant), for photon-limited data.

    `rate` is the model's expected photon count, `counts` the measured photons.
    """
    rate = jnp.clip(rate, eps)
    return jnp.mean(rate - counts * jnp.log(rate))


def total_variation(x: jax.Array) -> jax.Array:
    """Anisotropic total variation, a smoothness/edge-preserving regularizer."""
    tv = jnp.asarray(0.0)
    for ax in range(x.ndim):
        tv = tv + jnp.mean(jnp.abs(jnp.diff(x, axis=ax)))
    return tv


def softplus_nonneg(raw: jax.Array) -> jax.Array:
    """Map an unconstrained array to a non-negative one (for densities/intensities)."""
    return jax.nn.softplus(raw)  # type: ignore[no-any-return]


class FitResult(NamedTuple):
    params: PyTree
    losses: list[float]


def fit(
    loss_fn: Callable[[PyTree], jax.Array],
    init_params: PyTree,
    *,
    steps: int = 500,
    learning_rate: float = 1e-2,
    optimizer: optax.GradientTransformation | None = None,
    callback: Callable[[int, float, PyTree], None] | None = None,
) -> FitResult:
    """Minimize `loss_fn(params)` with Adam (or a provided optax optimizer).

    Returns the optimized params and the per-step loss history.
    """
    optimizer = optimizer or optax.adam(learning_rate)
    state = optimizer.init(init_params)
    params = init_params

    @jax.jit
    def step(
        params: PyTree, state: optax.OptState
    ) -> tuple[PyTree, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    losses: list[float] = []
    for i in range(steps):
        params, state, loss = step(params, state)
        losses.append(float(loss))
        if callback is not None:
            callback(i, losses[-1], params)
    return FitResult(params, losses)


def zernike_coeffs(
    ansi_indices: Sequence[int], init: float | Sequence[float] = 0.0
) -> jax.Array:
    """Build an initial Zernike-coefficient vector (the optimization variable)."""
    n = len(tuple(ansi_indices))
    if isinstance(init, int | float):
        return jnp.full((n,), float(init))
    return jnp.asarray(init, dtype=float)
