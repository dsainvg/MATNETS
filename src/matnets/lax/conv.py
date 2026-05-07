"""Convolution-like structural matrix primitives."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array

from matnets._params import MatrixParams

Padding = Literal["VALID", "SAME"]


def _check_bias_shape(params: MatrixParams, q: int, n: int) -> None:
    if params.B.shape != (q, n, n):
        msg = f"bias must be shaped ({q}, {n}, {n})"
        raise ValueError(msg)


def _check_square_matrix_kernel(W: Array, *, rank: int, shape_name: str) -> None:
    if W.ndim != rank:
        msg = f"{shape_name} expects weights shaped with rank {rank}"
        raise ValueError(msg)
    if W.shape[-2] != W.shape[-1]:
        msg = (
            "matrix convolution weights must map square matrices: "
            "last two axes must match"
        )
        raise ValueError(msg)


def _positive_stride(stride: int) -> int:
    if stride <= 0:
        msg = "stride must be positive"
        raise ValueError(msg)
    return stride


def _same_padding(size: int) -> tuple[int, int]:
    left = (size - 1) // 2
    right = size - 1 - left
    return left, right


def matrix_conv1d(
    params: MatrixParams,
    x: Array,
    *,
    stride: int = 1,
    padding: Padding = "VALID",
) -> Array:
    """Slide matrix kernels across a sequence.

    ``params.W`` is shaped ``(q, p, r, n, n)`` and ``x`` is shaped
    ``(t, p, n, n)``. The output is ``(t_out, q, n, n)``.
    """

    stride = _positive_stride(stride)
    W = params.W
    _check_square_matrix_kernel(W, rank=5, shape_name="matrix_conv1d")

    q, p, kernel, n, _ = W.shape
    x = jnp.asarray(x)
    if x.ndim != 4:
        msg = "matrix_conv1d expects inputs shaped (t, p, n, n)"
        raise ValueError(msg)
    if x.shape[1:] != (p, n, n):
        msg = "input axes must match weight axes: x must be shaped (t, p, n, n)"
        raise ValueError(msg)
    _check_bias_shape(params, q, n)

    if padding == "SAME":
        pad_left, pad_right = _same_padding(kernel)
    elif padding == "VALID":
        pad_left, pad_right = 0, 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    x = jnp.pad(x, ((pad_left, pad_right), (0, 0), (0, 0), (0, 0)))
    out_t = (x.shape[0] - kernel) // stride + 1
    windows = jnp.stack(
        [x[offset : offset + out_t * stride : stride] for offset in range(kernel)],
        axis=1,
    )
    return jnp.einsum("qprak,trpkc->tqac", W, windows) + params.B


def matrix_conv2d(
    params: MatrixParams,
    x: Array,
    *,
    stride: int | tuple[int, int] = 1,
    padding: Padding = "VALID",
) -> Array:
    """Slide matrix kernels across a 2D grid.

    ``params.W`` is shaped ``(q, p, h, w, n, n)`` and ``x`` is shaped
    ``(y, x, p, n, n)``. The output is ``(y_out, x_out, q, n, n)``.
    """

    if isinstance(stride, int):
        stride_y = stride_x = _positive_stride(stride)
    else:
        stride_y = _positive_stride(stride[0])
        stride_x = _positive_stride(stride[1])

    W = params.W
    _check_square_matrix_kernel(W, rank=6, shape_name="matrix_conv2d")

    q, p, kernel_y, kernel_x, n, _ = W.shape
    x = jnp.asarray(x)
    if x.ndim != 5:
        msg = "matrix_conv2d expects inputs shaped (y, x, p, n, n)"
        raise ValueError(msg)
    if x.shape[2:] != (p, n, n):
        msg = "input axes must match weight axes: x must be shaped (y, x, p, n, n)"
        raise ValueError(msg)
    _check_bias_shape(params, q, n)

    if padding == "SAME":
        pad_top, pad_bottom = _same_padding(kernel_y)
        pad_left, pad_right = _same_padding(kernel_x)
    elif padding == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    x = jnp.pad(
        x,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
    )
    out_y = (x.shape[0] - kernel_y) // stride_y + 1
    out_x = (x.shape[1] - kernel_x) // stride_x + 1
    rows = []
    for y_offset in range(kernel_y):
        cols = []
        for x_offset in range(kernel_x):
            cols.append(
                x[
                    y_offset : y_offset + out_y * stride_y : stride_y,
                    x_offset : x_offset + out_x * stride_x : stride_x,
                ]
            )
        rows.append(jnp.stack(cols, axis=2))
    windows = jnp.stack(rows, axis=2)
    return jnp.einsum("qphwak,ijhwpkc->ijqac", W, windows) + params.B
