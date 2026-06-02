import jax
import jax.nn as jnn
import jax.numpy as jnp

from ..__utils import _safe_det_root

sigmoid = jnn.sigmoid
tanh = jnn.tanh
softplus = jnn.softplus


def sigmoidd(x: jax.Array) -> jax.Array:
    """Determinant-gated sigmoid scaling.

    For each matrix, computes ``sigmoid(det(X)^(1/n)) / det(X)^(1/n)`` and
    multiplies every element of the matrix by that scalar.
    """
    dets = jnp.linalg.det(x)
    root = _safe_det_root(dets, x.shape[-1])
    scale = jnn.sigmoid(root) / root
    return x * scale[..., jnp.newaxis, jnp.newaxis]


def tanhd(x: jax.Array) -> jax.Array:
    """Determinant-gated tanh scaling.

    For each matrix, computes ``tanh(det(X)^(1/n)) / det(X)^(1/n)`` and
    multiplies every element of the matrix by that scalar.
    """
    dets = jnp.linalg.det(x)
    root = _safe_det_root(dets, x.shape[-1])
    scale = jnp.tanh(root) / root
    return x * scale[..., jnp.newaxis, jnp.newaxis]


def softplusd(x: jax.Array) -> jax.Array:
    """Determinant-gated softplus scaling.

    For each matrix, computes ``softplus(det(X)^(1/n)) / det(X)^(1/n)`` and
    multiplies every element of the matrix by that scalar.
    """
    dets = jnp.linalg.det(x)
    root = _safe_det_root(dets, x.shape[-1])
    scale = jnn.softplus(root) / root
    return x * scale[..., jnp.newaxis, jnp.newaxis]


def sst(x: jax.Array) -> jax.Array:
    """Scaled squared tanh activation.

    Applies element-wise tanh, squares the resulting matrices, and
    scales them by n^-2 where n is the matrix dimension.
    """
    n = x.shape[-1]
    t = jnp.tanh(x)
    s = jnp.matmul(t, t)
    scale = float(n) ** -2
    return s * scale


def sss(x: jax.Array) -> jax.Array:
    """Scaled squared sigmoid activation.

    Applies element-wise sigmoid, squares the resulting matrices, and
    scales them by n^-2 where n is the matrix dimension.
    """
    n = x.shape[-1]
    t = jnn.sigmoid(x)
    s = jnp.matmul(t, t)
    scale = float(n) ** -2
    return s * scale
