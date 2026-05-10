"""Data preprocessing utilities for MATNETS."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view


def _interleaved_axis_order(size: int, n: int) -> npt.NDArray[Any]:
    """Generate an interleaved permutation order for an axis."""
    block_end = (size // n) * n
    order = np.arange(block_end).reshape(-1, n).T.reshape(-1)
    if block_end < size:
        order = np.concatenate((order, np.arange(block_end, size)))
    return order


def embed_pixels(
    imgs: npt.NDArray[Any],
    n: int,
    spatial_axes: int | tuple[int, ...] = (1, 2),
    interleave: bool | tuple[bool, ...] = False,
) -> npt.NDArray[Any]:
    """Extract n x n local neighborhood padded with zeros.

    Transforms spatial dimensions by extracting local patches. If `interleave` is
    set, the spatial axes are permuted to interleave the extracted blocks.

    Args:
        imgs: Input array (e.g., shape `(N, H, W, C)`).
        n: The matrix size (patch size) to extract along each spatial axis.
        spatial_axes: The axes over which to extract patches. Defaults to `(1, 2)`.
        interleave: Whether to interleave the spatial axes. Can be a single boolean
            or a tuple of booleans matching the length of `spatial_axes`.

    Returns:
        Array with new window dimensions appended.
        For `(N, H, W, C)` with `spatial_axes=(1, 2)`, the output is
        `(N, H, W, C, n, n)`.
    """
    imgs = np.asarray(imgs)

    if isinstance(spatial_axes, int):
        spatial_axes = (spatial_axes,)

    if isinstance(interleave, bool):
        interleave = (interleave,) * len(spatial_axes)

    if len(spatial_axes) != len(interleave):
        msg = "interleave must be a boolean or a tuple matching spatial_axes length"
        raise ValueError(msg)

    # Normalize axes to be positive indices
    spatial_axes = tuple(ax % imgs.ndim for ax in spatial_axes)

    pad_width = [(0, 0)] * imgs.ndim
    pad_top = n // 2
    pad_bot = (n - 1) // 2

    for ax in spatial_axes:
        pad_width[ax] = (pad_top, pad_bot)

    padded = np.pad(imgs, pad_width, mode="constant")
    window_shape = (n,) * len(spatial_axes)

    windows = sliding_window_view(padded, window_shape, axis=spatial_axes)  # type: ignore[call-overload]

    for ax, inter in zip(spatial_axes, interleave, strict=False):
        if inter:
            order = _interleaved_axis_order(windows.shape[ax], n)
            windows = np.take(windows, order, axis=ax)

    return cast(npt.NDArray[Any], windows.copy())
