"""Core dense matrix primitive."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from matnets._params import MatrixParams


def identity(x: Array) -> Array:
    return x


@jax.custom_jvp
def _dense_linear(W: Array, B: Array, x: Array) -> Array:
    return jnp.einsum("qpak,pkc->qac", W, x) + B


@_dense_linear.defjvp
def _dense_linear_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
) -> tuple[Array, Array]:
    W, B, x = primals
    dW, dB, dx = tangents
    y = _dense_linear(W, B, x)
    dy = jnp.einsum("qpak,pkc->qac", dW, x)
    dy = dy + jnp.einsum("qpak,pkc->qac", W, dx) + dB
    return y, dy


def dense(
    params: MatrixParams,
    x: Array,
    activation: Callable[[Array], Array] = identity,
) -> Array:
    """Apply the core square-matrix contraction ``qpak,pkc -> qac``."""

    W = params.W
    x = jnp.asarray(x)
    if W.ndim != 4:
        msg = "dense expects weights shaped (q, p, n, n)"
        raise ValueError(msg)

    q, p, n, k = W.shape
    if n != k:
        msg = "dense weights must map square matrices: last two axes must match"
        raise ValueError(msg)
    if x.shape != (p, n, n):
        msg = f"dense expects input shaped ({p}, {n}, {n})"
        raise ValueError(msg)
    if params.B.shape != (q, n, n):
        msg = f"dense bias must be shaped ({q}, {n}, {n})"
        raise ValueError(msg)

    return activation(_dense_linear(W, params.B, x))
