"""JAX primitives for matrix-valued neural networks."""

from matnets import lax, nn
from matnets._dense import dense
from matnets._params import MatrixParams, init

__all__ = ["MatrixParams", "dense", "init", "lax", "nn"]

__version__ = "0.1.0"
