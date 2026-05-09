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
    q, p, a, k = W.shape
    c = x.shape[-1]
    W_flat = jnp.reshape(jnp.transpose(W, (0, 2, 1, 3)), (q * a, p * k))
    x_flat = jnp.reshape(x, (p * k, c))
    out = jnp.matmul(W_flat, x_flat)
    return jnp.reshape(out, (q, a, c)) + B


@_dense_linear.defjvp
def _dense_linear_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
) -> tuple[Array, Array]:
    W, B, x = primals
    dW, dB, dx = tangents
    q, p, a, k = W.shape
    c = x.shape[-1]

    W_flat = jnp.reshape(jnp.transpose(W, (0, 2, 1, 3)), (q * a, p * k))
    dW_flat = jnp.reshape(jnp.transpose(dW, (0, 2, 1, 3)), (q * a, p * k))
    x_flat = jnp.reshape(x, (p * k, c))
    dx_flat = jnp.reshape(dx, (p * k, c))

    y_flat = jnp.matmul(W_flat, x_flat)
    y = jnp.reshape(y_flat, (q, a, c)) + B

    dy_flat = jnp.matmul(dW_flat, x_flat) + jnp.matmul(W_flat, dx_flat)
    dy = jnp.reshape(dy_flat, (q, a, c)) + dB

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
