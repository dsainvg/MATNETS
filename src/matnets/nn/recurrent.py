"""Recurrent patterns built from the dense matrix primitive."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jax import Array

from matnets._dense import dense
from matnets._params import MatrixParams


def rnn_step(
    params: MatrixParams,
    carry: Array,
    x: Array,
    *,
    activation: Callable[[Array], Array] = jnp.tanh,
) -> tuple[Array, Array]:
    """Simple RNN step suitable for use inside ``jax.lax.scan``."""

    combined = jnp.concatenate([carry, x], axis=0)
    next_carry = dense(params, combined, activation)
    return next_carry, next_carry


def lstm_step(
    params: Mapping[str, MatrixParams],
    carry: tuple[Array, Array],
    x: Array,
) -> tuple[tuple[Array, Array], Array]:
    """LSTM step using ``i``, ``f``, ``g``, and ``o`` dense gate params."""

    h, c = carry
    combined = jnp.concatenate([h, x], axis=0)
    i = dense(params["i"], combined, jax.nn.sigmoid)
    f = dense(params["f"], combined, jax.nn.sigmoid)
    g = dense(params["g"], combined, jnp.tanh)
    o = dense(params["o"], combined, jax.nn.sigmoid)
    next_c = f * c + i * g
    next_h = o * jnp.tanh(next_c)
    return (next_h, next_c), next_h


def gru_step(
    params: Mapping[str, MatrixParams],
    carry: Array,
    x: Array,
) -> tuple[Array, Array]:
    """GRU step using ``z``, ``r``, and ``n`` dense gate params."""

    combined = jnp.concatenate([carry, x], axis=0)
    z = dense(params["z"], combined, jax.nn.sigmoid)
    r = dense(params["r"], combined, jax.nn.sigmoid)
    candidate_input = jnp.concatenate([r * carry, x], axis=0)
    n = dense(params["n"], candidate_input, jnp.tanh)
    next_carry = (1.0 - z) * n + z * carry
    return next_carry, next_carry
