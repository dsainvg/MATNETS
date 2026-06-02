import numpy as np
import pytest

from matnets.utils import embed_pixels, embed_sequence


def test_embed_pixels_shape() -> None:
    imgs = np.zeros((2, 10, 10, 3))
    out = embed_pixels(imgs, n=3, spatial_axes=(1, 2))
    assert out.shape == (2, 10, 10, 3, 3, 3)


def test_embed_pixels_1d() -> None:
    seq = np.zeros((5, 20, 4))
    out = embed_pixels(seq, n=2, spatial_axes=1)
    assert out.shape == (5, 20, 4, 2)


def test_embed_pixels_interleave() -> None:
    imgs = np.arange(100).reshape((1, 10, 10, 1))

    out_normal = embed_pixels(imgs, n=2, spatial_axes=(1, 2), interleave=False)
    out_interleaved = embed_pixels(imgs, n=2, spatial_axes=(1, 2), interleave=True)

    assert out_normal.shape == out_interleaved.shape
    assert not np.array_equal(out_normal, out_interleaved)

    # Check if interleaving logic matches _interleaved_axis_order expected behavior
    # For size=10, n=2: block_end=10. np.arange(10).reshape(-1, 2).T.reshape(-1)
    # This interleaves the groups. Elements: 0, 2, 4, 6, 8, 1, 3, 5, 7, 9
    expected_order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    expected_slice = out_normal[:, expected_order][:, :, expected_order]

    assert np.array_equal(out_interleaved, expected_slice)


def test_embed_pixels_validation() -> None:
    imgs = np.zeros((2, 10, 10, 3))
    with pytest.raises(
        ValueError, match="interleave must be a boolean or a tuple matching"
    ):
        embed_pixels(imgs, n=3, spatial_axes=(1, 2), interleave=(True,))


def test_embed_sequence_shape() -> None:
    # 2D: (N, T) -> (N, T, n, n)
    seq_nt = np.zeros((2, 10))
    out_nt = embed_sequence(seq_nt, n=3, axis=1)
    assert out_nt.shape == (2, 10, 3, 3)

    # 1D: (T,) -> (T, n, n)
    seq_t = np.zeros((10,))
    out_t = embed_sequence(seq_t, n=3, axis=0)
    assert out_t.shape == (10, 3, 3)

    # 3D: (N, T, C) -> (N, T, C, n, n)
    seq_ntc = np.zeros((2, 10, 5))
    out_ntc = embed_sequence(seq_ntc, n=3, axis=1)
    assert out_ntc.shape == (2, 10, 5, 3, 3)

def test_embed_sequence_channels() -> None:
    # 5 time steps, 2 channels
    seq = np.arange(1, 11).reshape(5, 2)
    out = embed_sequence(seq, n=3, axis=0)
    assert out.shape == (5, 2, 3, 3)

    # Check the result for the last time step (t=4)
    # Channel 0: [9, 7, 5] (current=9, t-1=7, t-2=5)
    # Channel 1: [10, 8, 6] (current=10, t-1=8, t-2=6)
    expected_c0 = np.array([
        [9, 7, 5],
        [7, 9, 7],
        [5, 7, 9]
    ])
    expected_c1 = np.array([
        [10, 8, 6],
        [8, 10, 8],
        [6, 8, 10]
    ])
    assert np.array_equal(out[4, 0], expected_c0)
    assert np.array_equal(out[4, 1], expected_c1)


def test_embed_sequence_values() -> None:
    # 1D sequence: 1, 2, 3, 4, 5
    seq = np.arange(1, 6)
    out = embed_sequence(seq, n=3, axis=0)

    # At t=0, x_0=1, padded with 0s for past:
    expected_t0 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    assert np.array_equal(out[0], expected_t0)

    # At t=4, x_4=5, x_3=4, x_2=3:
    expected_t4 = np.array([
        [5, 4, 3],
        [4, 5, 4],
        [3, 4, 5]
    ])
    assert np.array_equal(out[4], expected_t4)

