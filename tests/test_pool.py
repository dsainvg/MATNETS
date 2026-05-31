import jax.numpy as jnp
import pytest

from matnets.conv import (
    avg_pool1d,
    avgd_pool1d,
    avgd_pool2d,
    max_pool1d,
    max_pool2d,
    sum_pool1d,
    sum_pool2d,
)


def test_avg_pool1d_slides_window() -> None:
    x = jnp.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])
    out = avg_pool1d(x, window_size=2)
    assert out.shape == (3, 1, 1, 1)
    assert out[:, 0, 0, 0].tolist() == pytest.approx([1.5, 2.5, 3.5])


def test_avgd_pool1d_slides_window() -> None:
    x = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]], [[[2.0, 0.0], [0.0, 2.0]]]])
    out = avgd_pool1d(x, window_size=2)
    assert out.shape == (1, 1, 2, 2)
    # Under 1/det^(1/n) scaling:
    # M1 = I, det = 1, root = 1, weighted M1 = I
    # M2 = 2I, det = 4, root = 2, weighted M2 = I
    # Sum = 2I
    assert jnp.allclose(out[0, 0], jnp.array([[2.0, 0.0], [0.0, 2.0]]))


def test_avgd_pool2d_slides_window() -> None:
    # 2D inputs of shape (y, x, p, n, n)
    x = jnp.array([
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        [[[[2.0, 0.0], [0.0, 2.0]]]],
    ]) # Shape: (2, 1, 1, 2, 2)
    out = avgd_pool2d(x, window_size=(2, 1))
    assert out.shape == (1, 1, 1, 2, 2)
    # Sum = 2I
    assert jnp.allclose(out[0, 0, 0], jnp.array([[2.0, 0.0], [0.0, 2.0]]))


def test_max_pool1d_slides_window() -> None:
    x = jnp.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])

    out = max_pool1d(x, window_size=2)
    assert out.shape == (3, 1, 1, 1)
    assert out[:, 0, 0, 0].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_sum_pool1d_slides_window() -> None:
    x = jnp.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])

    out = sum_pool1d(x, window_size=2)
    assert out.shape == (3, 1, 1, 1)
    assert out[:, 0, 0, 0].tolist() == pytest.approx([3.0, 5.0, 7.0])


def test_max_pool2d_slides_window() -> None:
    x = jnp.array(
        [
            [[[[1.0]]], [[[2.0]]]],
            [[[[3.0]]], [[[4.0]]]],
        ]
    )
    out = max_pool2d(x, window_size=2)
    assert out.shape == (1, 1, 1, 1, 1)
    assert out[0, 0, 0, 0, 0] == pytest.approx(4.0)


def test_sum_pool2d_slides_window() -> None:
    x = jnp.array(
        [
            [[[[1.0]]], [[[2.0]]]],
            [[[[3.0]]], [[[4.0]]]],
        ]
    )
    out = sum_pool2d(x, window_size=2)
    assert out.shape == (1, 1, 1, 1, 1)
    assert out[0, 0, 0, 0, 0] == pytest.approx(10.0)


def test_pool1d_rejects_bad_shapes() -> None:
    x = jnp.ones((2, 1, 2, 3))
    with pytest.raises(ValueError, match="square matrices"):
        max_pool1d(x, window_size=2)

    x2 = jnp.ones((2, 1, 2))
    with pytest.raises(ValueError, match=r"rank 4 \(unbatched\) or 5 \(batched\)"):
        sum_pool1d(x2, window_size=2)


def test_pool2d_rejects_bad_shapes() -> None:
    x = jnp.ones((2, 2, 1, 2, 3))
    with pytest.raises(ValueError, match="square matrices"):
        max_pool2d(x, window_size=2)

    x2 = jnp.ones((2, 2, 1, 2))
    with pytest.raises(ValueError, match=r"rank 5 \(unbatched\) or 6 \(batched\)"):
        sum_pool2d(x2, window_size=2)


def test_pool_supports_native_batching() -> None:
    x_1d_batch = jnp.ones((2, 3, 1, 1, 1))
    out_1d = max_pool1d(x_1d_batch, window_size=2)
    assert out_1d.shape == (2, 2, 1, 1, 1)

    x_2d_batch = jnp.ones((2, 2, 2, 1, 1, 1))
    out_2d = sum_pool2d(x_2d_batch, window_size=2)
    assert out_2d.shape == (2, 1, 1, 1, 1, 1)
