"""JAX-level structural primitives."""

from matnets.lax.attention import (
    frobenius_score,
    matrix_attention,
    scaled_frobenius_score,
)
from matnets.lax.conv import matrix_conv1d, matrix_conv2d

__all__ = [
    "frobenius_score",
    "matrix_attention",
    "matrix_conv1d",
    "matrix_conv2d",
    "scaled_frobenius_score",
]
