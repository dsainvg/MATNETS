"""Pooling layer primitives."""

from matnets.conv.pool import (
    avg_pool1d,
    avg_pool2d,
    avgd_pool1d,
    avgd_pool2d,
    max_pool1d,
    max_pool2d,
    maxd_pool1d,
    maxd_pool2d,
    sum_pool1d,
    sum_pool2d,
)

__all__ = [
    "avg_pool1d",
    "avg_pool2d",
    "avgd_pool1d",
    "avgd_pool2d",
    "max_pool1d",
    "max_pool2d",
    "maxd_pool1d",
    "maxd_pool2d",
    "sum_pool1d",
    "sum_pool2d",
]
