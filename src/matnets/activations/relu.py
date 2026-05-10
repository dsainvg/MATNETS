import jax
import jax.numpy as jnp


def relu(x: jax.Array) -> jax.Array:
    """Element-wise rectified linear unit."""
    return jnp.maximum(0, x)


def relud(x: jax.Array) -> jax.Array:
    """Rectified linear unit based on the determinant of the last two axes.

    If the determinant of the last two axes is positive, the input is returned
    unchanged. Otherwise, all elements are set to zero.

    Args:
        x: Input JAX array of shape (..., n, n).

    Returns:
        The activated array.
    """
    dets = jnp.linalg.det(x)
    return jnp.where(dets[..., jnp.newaxis, jnp.newaxis] > 0, x, 0.0)


def leaky_relud(x: jax.Array, negative_slope: float = 0.01) -> jax.Array:
    """Leaky rectified linear unit based on the determinant of the last two axes.

    If the determinant of the last two axes is positive, the input is returned
    unchanged. Otherwise, the matrix is scaled by `negative_slope`.

    Args:
        x: Input JAX array of shape (..., n, n).
        negative_slope: The slope for the non-positive determinant case.

    Returns:
        The activated array.
    """
    dets = jnp.linalg.det(x)
    return jnp.where(dets[..., jnp.newaxis, jnp.newaxis] > 0, x, negative_slope * x)
