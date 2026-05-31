import jax
import jax.numpy as jnp

_EPS = 1e-7


def _safe_det_root(dets: jax.Array, n: int) -> jax.Array:
    """Return ``sign(det) * |det|^(1/n)`` with near-zero values clamped."""
    abs_dets = jnp.maximum(jnp.abs(dets), _EPS)
    return jnp.sign(dets) * abs_dets ** (1.0 / n)
