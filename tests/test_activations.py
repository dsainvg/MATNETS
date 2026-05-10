import jax.numpy as jnp

from matnets.activations import (
    elud,
    leaky_relud,
    relud,
)


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


def test_elud_branches_on_determinant() -> None:
    # det = 1
    x_pos = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    # det = -1
    x_neg = jnp.array([[0.5, 0.0], [0.0, -2.0]])

    out_pos = elud(x_pos)
    out_neg = elud(x_neg)

    assert jnp.allclose(out_pos, x_pos)
    # ELU neg branch: expm(X) - I
    expected_neg = jnp.diag(jnp.array([jnp.exp(0.5) - 1.0, jnp.exp(-2.0) - 1.0]))
    assert jnp.allclose(out_neg, expected_neg)


def test_leaky_relud_scales_negative_determinant() -> None:
    # det = -1.0
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    slope = 0.1
    output = leaky_relud(x, negative_slope=slope)
    assert jnp.allclose(output, x * slope)
