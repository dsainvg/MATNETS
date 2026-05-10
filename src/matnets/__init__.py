"""JAX primitives for matrix-valued neural networks."""

from matnets import conv, lax, nn, utils
from matnets._dense import dense
from matnets._params import MatrixParams, init

__all__ = ["MatrixParams", "conv", "dense", "init", "lax", "nn", "utils"]

__version__ = "0.1.0"
