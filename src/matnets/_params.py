"""Parameter containers and initializers for matrix primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MatrixParams:
    """Weights and bias for matrix contractions.

    Dense weights use ``(q, p, n, n)`` and consume inputs shaped ``(p, n, n)``.
    The bias is shaped ``(q, n, n)`` so each output unit has a full matrix bias.
    Structural primitives may use higher-rank weights while keeping the same
    small, pytree-friendly container.
    """

    W: Array
    B: Array

    def tree_flatten(self) -> tuple[tuple[Array, Array], None]:
        return (self.W, self.B), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Array, Array],
    ) -> MatrixParams:
        del aux_data
        return cls(*children)


def init(
    key: Array,
    p: int,
    q: int,
    n: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> MatrixParams:
    """Initialize dense matrix parameters with Glorot-uniform weights."""

    if p <= 0 or q <= 0 or n <= 0:
        msg = "p, q, and n must all be positive"
        raise ValueError(msg)

    fan_in = p * n
    fan_out = q * n
    limit = sqrt(6.0 / (fan_in + fan_out))
    W = jax.random.uniform(
        key,
        shape=(q, p, n, n),
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )
    B = jnp.zeros((q, n, n), dtype=dtype)
    return MatrixParams(W=W, B=B)
