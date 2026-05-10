"""Pooling structural matrix primitives."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from matnets.lax.conv import Padding, _positive_stride, _same_padding


def _check_pool_input(x: Array, expected_rank: int, shape_name: str) -> None:
    if x.ndim != expected_rank and x.ndim != expected_rank + 1:
        msg = (
            f"{shape_name} expects inputs shaped with rank {expected_rank} "
            f"(unbatched) or {expected_rank + 1} (batched)"
        )
        raise ValueError(msg)
    if x.shape[-2] != x.shape[-1]:
        msg = f"{shape_name} expects square matrices: last two axes must match"
        raise ValueError(msg)


def max_pool1d(
    x: Array,
    *,
    window_size: int,
    stride: int = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply max pooling over a 1D sequence of matrices.

    ``x`` is shaped ``(t, p, n, n)``. The output is ``(t_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 5:
        return jax.vmap(
            lambda x_: max_pool1d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 4, "max_pool1d")
    stride = _positive_stride(stride)

    if padding == "SAME":
        pad_left, pad_right = _same_padding(window_size)
    elif padding == "VALID":
        pad_left, pad_right = 0, 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    return cast(
        Array,
        jax.lax.reduce_window(
            x,
            jnp.array(-jnp.inf, dtype=x.dtype),
            jax.lax.max,
            window_dimensions=(window_size, 1, 1, 1),
            window_strides=(stride, 1, 1, 1),
            padding=((pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
        ),
    )


def sum_pool1d(
    x: Array,
    *,
    window_size: int,
    stride: int = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply sum pooling over a 1D sequence of matrices.

    ``x`` is shaped ``(t, p, n, n)``. The output is ``(t_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 5:
        return jax.vmap(
            lambda x_: sum_pool1d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 4, "sum_pool1d")
    stride = _positive_stride(stride)

    if padding == "SAME":
        pad_left, pad_right = _same_padding(window_size)
    elif padding == "VALID":
        pad_left, pad_right = 0, 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    return cast(
        Array,
        jax.lax.reduce_window(
            x,
            jnp.array(0.0, dtype=x.dtype),
            jax.lax.add,
            window_dimensions=(window_size, 1, 1, 1),
            window_strides=(stride, 1, 1, 1),
            padding=((pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
        ),
    )


def max_pool2d(
    x: Array,
    *,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply max pooling over a 2D grid of matrices.

    ``x`` is shaped ``(y, x, p, n, n)``. The output is ``(y_out, x_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 6:
        return jax.vmap(
            lambda x_: max_pool2d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 5, "max_pool2d")

    if isinstance(window_size, int):
        wy = wx = window_size
    else:
        wy, wx = window_size

    if isinstance(stride, int):
        sy = sx = _positive_stride(stride)
    else:
        sy = _positive_stride(stride[0])
        sx = _positive_stride(stride[1])

    if padding == "SAME":
        pad_top, pad_bottom = _same_padding(wy)
        pad_left, pad_right = _same_padding(wx)
    elif padding == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    return cast(
        Array,
        jax.lax.reduce_window(
            x,
            jnp.array(-jnp.inf, dtype=x.dtype),
            jax.lax.max,
            window_dimensions=(wy, wx, 1, 1, 1),
            window_strides=(sy, sx, 1, 1, 1),
            padding=(
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
                (0, 0),
                (0, 0),
            ),
        ),
    )


def sum_pool2d(
    x: Array,
    *,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply sum pooling over a 2D grid of matrices.

    ``x`` is shaped ``(y, x, p, n, n)``. The output is ``(y_out, x_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 6:
        return jax.vmap(
            lambda x_: sum_pool2d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 5, "sum_pool2d")

    if isinstance(window_size, int):
        wy = wx = window_size
    else:
        wy, wx = window_size

    if isinstance(stride, int):
        sy = sx = _positive_stride(stride)
    else:
        sy = _positive_stride(stride[0])
        sx = _positive_stride(stride[1])

    if padding == "SAME":
        pad_top, pad_bottom = _same_padding(wy)
        pad_left, pad_right = _same_padding(wx)
    elif padding == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    return cast(
        Array,
        jax.lax.reduce_window(
            x,
            jnp.array(0.0, dtype=x.dtype),
            jax.lax.add,
            window_dimensions=(wy, wx, 1, 1, 1),
            window_strides=(sy, sx, 1, 1, 1),
            padding=(
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
                (0, 0),
                (0, 0),
            ),
        ),
    )


def maxd_pool1d(
    x: Array,
    *,
    window_size: int,
    stride: int = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply max determinant pooling over a 1D sequence of matrices.

    ``x`` is shaped ``(t, p, n, n)``. The output is ``(t_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 5:
        return jax.vmap(
            lambda x_: maxd_pool1d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 4, "maxd_pool1d")
    stride = _positive_stride(stride)

    if padding == "SAME":
        pad_left, pad_right = _same_padding(window_size)
    elif padding == "VALID":
        pad_left, pad_right = 0, 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    x_padded = jnp.pad(x, ((pad_left, pad_right), (0, 0), (0, 0), (0, 0)))
    t_padded, p, n, _ = x_padded.shape
    t_out = (t_padded - window_size) // stride + 1

    def get_window(i: Array) -> Array:
        start = i * stride
        return jax.lax.dynamic_slice(x_padded, (start, 0, 0, 0), (window_size, p, n, n))

    windows = jax.vmap(get_window)(jnp.arange(t_out))
    dets = jnp.linalg.det(windows)
    max_indices = jnp.argmax(dets, axis=1)

    def gather_fn(window: Array, max_idx: Array) -> Array:
        def gather_p(win_p: Array, idx: Array) -> Array:
            return win_p[idx]

        return jax.vmap(gather_p, in_axes=(1, 0))(window, max_idx)

    return jax.vmap(gather_fn)(windows, max_indices)


def maxd_pool2d(
    x: Array,
    *,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply max determinant pooling over a 2D grid of matrices.

    ``x`` is shaped ``(y, x, p, n, n)``. The output is ``(y_out, x_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 6:
        return jax.vmap(
            lambda x_: maxd_pool2d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 5, "maxd_pool2d")

    if isinstance(window_size, int):
        wy = wx = window_size
    else:
        wy, wx = window_size

    if isinstance(stride, int):
        sy = sx = _positive_stride(stride)
    else:
        sy = _positive_stride(stride[0])
        sx = _positive_stride(stride[1])

    if padding == "SAME":
        pad_top, pad_bottom = _same_padding(wy)
        pad_left, pad_right = _same_padding(wx)
    elif padding == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    x_padded = jnp.pad(
        x, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0), (0, 0))
    )
    y_padded, x_padded_dim, p, n, _ = x_padded.shape
    y_out = (y_padded - wy) // sy + 1
    x_out = (x_padded_dim - wx) // sx + 1

    def get_window(iy: Array, ix: Array) -> Array:
        start_y = iy * sy
        start_x = ix * sx
        return jax.lax.dynamic_slice(
            x_padded, (start_y, start_x, 0, 0, 0), (wy, wx, p, n, n)
        )

    grid_y, grid_x = jnp.meshgrid(jnp.arange(y_out), jnp.arange(x_out), indexing="ij")
    windows = jax.vmap(jax.vmap(get_window))(grid_y, grid_x)

    windows_flat = windows.reshape((y_out, x_out, wy * wx, p, n, n))
    dets = jnp.linalg.det(windows_flat)
    max_indices = jnp.argmax(dets, axis=2)

    def gather_fn(window_flat: Array, max_idx: Array) -> Array:
        def gather_p(win_p: Array, idx: Array) -> Array:
            return win_p[idx]

        return jax.vmap(gather_p, in_axes=(1, 0))(window_flat, max_idx)

    return jax.vmap(jax.vmap(gather_fn))(windows_flat, max_indices)


def avg_pool1d(
    x: Array,
    *,
    window_size: int,
    stride: int = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply average pooling over a 1D sequence of matrices.

    ``x`` is shaped ``(t, p, n, n)``. The output is ``(t_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 5:
        return jax.vmap(
            lambda x_: avg_pool1d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 4, "avg_pool1d")
    stride = _positive_stride(stride)

    if padding == "SAME":
        pad_left, pad_right = _same_padding(window_size)
    elif padding == "VALID":
        pad_left, pad_right = 0, 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    summed = cast(
        Array,
        jax.lax.reduce_window(
            x,
            jnp.array(0.0, dtype=x.dtype),
            jax.lax.add,
            window_dimensions=(window_size, 1, 1, 1),
            window_strides=(stride, 1, 1, 1),
            padding=((pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
        ),
    )
    return summed / window_size


def avg_pool2d(
    x: Array,
    *,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply average pooling over a 2D grid of matrices.

    ``x`` is shaped ``(y, x, p, n, n)``. The output is ``(y_out, x_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 6:
        return jax.vmap(
            lambda x_: avg_pool2d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 5, "avg_pool2d")

    if isinstance(window_size, int):
        wy = wx = window_size
    else:
        wy, wx = window_size

    if isinstance(stride, int):
        sy = sx = _positive_stride(stride)
    else:
        sy = _positive_stride(stride[0])
        sx = _positive_stride(stride[1])

    if padding == "SAME":
        pad_top, pad_bottom = _same_padding(wy)
        pad_left, pad_right = _same_padding(wx)
    elif padding == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    summed = cast(
        Array,
        jax.lax.reduce_window(
            x,
            jnp.array(0.0, dtype=x.dtype),
            jax.lax.add,
            window_dimensions=(wy, wx, 1, 1, 1),
            window_strides=(sy, sx, 1, 1, 1),
            padding=(
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
                (0, 0),
                (0, 0),
            ),
        ),
    )
    return summed / (wy * wx)


def avgd_pool1d(
    x: Array,
    *,
    window_size: int,
    stride: int = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply average determinant-weighted pooling over a 1D sequence of matrices.

    ``x`` is shaped ``(t, p, n, n)``. The output is ``(t_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 5:
        return jax.vmap(
            lambda x_: avgd_pool1d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 4, "avgd_pool1d")
    stride = _positive_stride(stride)

    if padding == "SAME":
        pad_left, pad_right = _same_padding(window_size)
    elif padding == "VALID":
        pad_left, pad_right = 0, 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    x_padded = jnp.pad(x, ((pad_left, pad_right), (0, 0), (0, 0), (0, 0)))
    t_padded, p, n, _ = x_padded.shape
    t_out = (t_padded - window_size) // stride + 1

    def get_window(i: Array) -> Array:
        start = i * stride
        return jax.lax.dynamic_slice(x_padded, (start, 0, 0, 0), (window_size, p, n, n))

    windows = jax.vmap(get_window)(jnp.arange(t_out))
    dets = jnp.linalg.det(windows)
    weighted = windows / dets[..., None, None]
    return jnp.sum(weighted, axis=1)


def avgd_pool2d(
    x: Array,
    *,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: Padding = "VALID",
) -> Array:
    """Apply average determinant-weighted pooling over a 2D grid of matrices.

    ``x`` is shaped ``(y, x, p, n, n)``. The output is ``(y_out, x_out, p, n, n)``.
    """
    x = jnp.asarray(x)
    if x.ndim == 6:
        return jax.vmap(
            lambda x_: avgd_pool2d(
                x_, window_size=window_size, stride=stride, padding=padding
            )
        )(x)
    _check_pool_input(x, 5, "avgd_pool2d")

    if isinstance(window_size, int):
        wy = wx = window_size
    else:
        wy, wx = window_size

    if isinstance(stride, int):
        sy = sx = _positive_stride(stride)
    else:
        sy = _positive_stride(stride[0])
        sx = _positive_stride(stride[1])

    if padding == "SAME":
        pad_top, pad_bottom = _same_padding(wy)
        pad_left, pad_right = _same_padding(wx)
    elif padding == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        msg = "padding must be 'VALID' or 'SAME'"
        raise ValueError(msg)

    x_padded = jnp.pad(
        x, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0), (0, 0))
    )
    y_padded, x_padded_dim, p, n, _ = x_padded.shape
    y_out = (y_padded - wy) // sy + 1
    x_out = (x_padded_dim - wx) // sx + 1

    def get_window(iy: Array, ix: Array) -> Array:
        start_y = iy * sy
        start_x = ix * sx
        return jax.lax.dynamic_slice(
            x_padded, (start_y, start_x, 0, 0, 0), (wy, wx, p, n, n)
        )

    grid_y, grid_x = jnp.meshgrid(jnp.arange(y_out), jnp.arange(x_out), indexing="ij")
    windows = jax.vmap(jax.vmap(get_window))(grid_y, grid_x)

    windows_flat = windows.reshape((y_out, x_out, wy * wx, p, n, n))
    dets = jnp.linalg.det(windows_flat)
    weighted = windows_flat / dets[..., None, None]
    return jnp.sum(weighted, axis=2)
