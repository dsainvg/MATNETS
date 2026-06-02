"""JAX primitives for matrix-valued neural networks."""

from importlib import metadata

from matnets import activations, conv, lax, nn, utils
from matnets._dense import dense
from matnets._params import MatrixParams, init

__all__ = ["MatrixParams", "activations", "conv", "dense", "init", "lax", "nn", "utils"]

__version__ = metadata.version("matnets")
