"""Attention primitive for matrix-valued tokens."""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt

import jax
import jax.numpy as jnp
from jax import Array

from matnets._dense import dense
from matnets._params import MatrixParams

ScoreFn = Callable[[Array, Array], Array]


def frobenius_score(Q: Array, K: Array) -> Array:
    """Scalar Frobenius inner product between two matrix-valued tokens."""

    return jnp.einsum("pkc,pkc->", Q, K)


def scaled_frobenius_score(Q: Array, K: Array) -> Array:
    scale = sqrt(Q.shape[0] * Q.shape[1] * Q.shape[2])
    return frobenius_score(Q, K) / scale


def matrix_attention(
    params: MatrixParams | None,
    Q: Array,
    K: Array,
    V: Array,
    score_fn: ScoreFn = scaled_frobenius_score,
) -> Array:
    """Score, normalize, aggregate, and optionally project matrix tokens.

    ``Q``, ``K``, and ``V`` are token sequences shaped ``(t, p, n, n)``.
    ``score_fn`` defines the pairwise semantics and must return a scalar.
    Passing ``params=None`` returns the aggregated context directly; otherwise
    each context token is projected through ``dense(params, token)``.
    """

    Q = jnp.asarray(Q)
    K = jnp.asarray(K)
    V = jnp.asarray(V)
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        msg = "matrix_attention expects Q, K, and V shaped (t, p, n, n)"
        raise ValueError(msg)
    if Q.shape[1:] != K.shape[1:] or Q.shape[1:] != V.shape[1:]:
        msg = "Q, K, and V must share neuron and matrix axes"
        raise ValueError(msg)
    if Q.shape[-2] != Q.shape[-1]:
        msg = "matrix_attention expects square matrix tokens"
        raise ValueError(msg)

    scores = jax.vmap(lambda q: jax.vmap(lambda k: score_fn(q, k))(K))(Q)
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("ij,jpkc->ipkc", weights, V)
    if params is None:
        return context
    return jax.vmap(lambda token: dense(params, token))(context)
