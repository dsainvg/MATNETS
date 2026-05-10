import numpy as np
import pytest

from matnets.utils import embed_pixels


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
