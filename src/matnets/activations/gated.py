import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.scipy.linalg import expm

from ..__utils import _safe_det_root

relu = jnn.relu
leaky_relu = jnn.leaky_relu
elu = jnn.elu


def relud(x: jax.Array) -> jax.Array:
    """Determinant-gated matrix ReLU.

    Returns the input matrix if its determinant is positive, otherwise zeros.
    """
    dets = jnp.linalg.det(x)
    return jnp.where(dets[..., jnp.newaxis, jnp.newaxis] > 0, x, 0.0)


def leaky_relud(x: jax.Array, negative_slope: float = 0.01) -> jax.Array:
    """Determinant-gated matrix leaky ReLU.

    Returns the input matrix if its determinant is positive, otherwise scales it
    by `negative_slope`.
    """
    dets = jnp.linalg.det(x)
    return jnp.where(dets[..., jnp.newaxis, jnp.newaxis] > 0, x, negative_slope * x)


def elu_powered(x: jax.Array, alpha: float = 1.0) -> jax.Array:
    """Determinant-gated matrix ELU.
    @param alpha: the scaling factor for the negative branch, as in standard ELU.
    Returns the input matrix if its determinant is positive, otherwise uses the
    matrix exponential branch: alpha * (expm(X) - I).
    NOTE: the matrix exponential is used SO it is so slow. Use with caution.
    """
    dets = jnp.linalg.det(x)
    n = x.shape[-1]
    Ident = jnp.eye(n)
    neg_branch = alpha * (expm(x) - Ident)
    # Cast to jax.Array to satisfy mypy's no-any-return check on jnp.where
    return jax.lax.convert_element_type(
        jnp.where(dets[..., jnp.newaxis, jnp.newaxis] > 0, x, neg_branch), x.dtype
    )


def elud(x: jax.Array, alpha: float = 1.0) -> jax.Array:
    """Determinant-gated ELU scaling.

    For each matrix, computes ``elu(det(X)^(1/n), alpha) / det(X)^(1/n)`` and
    multiplies every element of the matrix by that scalar.
    """
    dets = jnp.linalg.det(x)
    root = _safe_det_root(dets, x.shape[-1])
    scale = jnn.elu(root, alpha) / root
    return x * scale[..., jnp.newaxis, jnp.newaxis]
