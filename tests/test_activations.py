import jax.numpy as jnp

from matnets.activations import leaky_relud, relud


def test_relud_positive_determinant() -> None:
    # Identity matrix has det = 1.0 > 0
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    output = relud(x)
    assert jnp.allclose(output, x)


def test_relud_negative_determinant() -> None:
    # Swapped identity has det = -1.0 <= 0
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    output = relud(x)
    assert jnp.allclose(output, jnp.zeros_like(x))


def test_relud_zero_determinant() -> None:
    # Zero matrix or singular matrix has det = 0.0 <= 0
    x = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    output = relud(x)
    assert jnp.allclose(output, jnp.zeros_like(x))


def test_relud_batch_broadcasting() -> None:
    # Batch of matrices
    x = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],  # det = 1
            [[0.0, 1.0], [1.0, 0.0]],  # det = -1
            [[2.0, 0.0], [0.0, 2.0]],  # det = 4
        ]
    )
    expected = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    output = relud(x)
    assert jnp.allclose(output, expected)


def test_relud_high_dim_broadcasting() -> None:
    # (2, 2, 2, 2)
    x1 = jnp.eye(2)
    x2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    x = jnp.stack([jnp.stack([x1, x2]), jnp.stack([x2, x1])])

    output = relud(x)

    assert jnp.allclose(output[0, 0], x1)
    assert jnp.allclose(output[0, 1], jnp.zeros((2, 2)))
    assert jnp.allclose(output[1, 0], jnp.zeros((2, 2)))
    assert jnp.allclose(output[1, 1], x1)


def test_leaky_relud_scales_negative_determinant() -> None:
    # det = -1.0
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    slope = 0.1
    output = leaky_relud(x, negative_slope=slope)
    assert jnp.allclose(output, x * slope)


def test_leaky_relud_keeps_positive_determinant() -> None:
    # det = 1.0
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    output = leaky_relud(x)
    assert jnp.allclose(output, x)
