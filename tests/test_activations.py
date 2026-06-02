import jax.numpy as jnp
from jax.nn import sigmoid as jax_sigmoid
from jax.nn import softplus as jax_softplus

from matnets.activations import (
    elu_powered,
    elud,
    leaky_relud,
    relud,
    sigmoidd,
    softplusd,
    sss,
    sst,
    tanhd,
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


def test_elu_powered_branches_on_determinant() -> None:
    # det = 1
    x_pos = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    # det = -1
    x_neg = jnp.array([[0.5, 0.0], [0.0, -2.0]])

    out_pos = elu_powered(x_pos)
    out_neg = elu_powered(x_neg)

    assert jnp.allclose(out_pos, x_pos)
    # ELU neg branch: expm(X) - I
    expected_neg = jnp.diag(jnp.array([jnp.exp(0.5) - 1.0, jnp.exp(-2.0) - 1.0]))
    assert jnp.allclose(out_neg, expected_neg)


def test_elud_scales_on_determinant() -> None:
    # 2x2 matrix
    x = jnp.array([[2.0, 0.0], [0.0, 2.0]])  # det = 4, root = 2.0
    out = elud(x, alpha=1.0)
    # expected scale: elu(2.0, 1.0) / 2.0 = 2.0 / 2.0 = 1.0
    assert jnp.allclose(out, x)


def test_leaky_relud_scales_negative_determinant() -> None:
    # det = -1.0
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    slope = 0.1
    output = leaky_relud(x, negative_slope=slope)
    assert jnp.allclose(output, x * slope)


def test_sigmoidd_scaling() -> None:
    x = jnp.array([[4.0, 0.0], [0.0, 4.0]])  # det = 16, root = 4.0
    out = sigmoidd(x)
    # expected scale: sigmoid(4.0) / 4.0
    expected_scale = jax_sigmoid(4.0) / 4.0
    assert jnp.allclose(out, x * expected_scale)


def test_tanhd_scaling() -> None:
    x = jnp.array([[4.0, 0.0], [0.0, 4.0]])  # det = 16, root = 4.0
    out = tanhd(x)
    # expected scale: tanh(4.0) / 4.0
    expected_scale = jnp.tanh(4.0) / 4.0
    assert jnp.allclose(out, x * expected_scale)


def test_softplusd_scaling() -> None:
    x = jnp.array([[4.0, 0.0], [0.0, 4.0]])  # det = 16, root = 4.0
    out = softplusd(x)
    # expected scale: softplus(4.0) / 4.0
    expected_scale = jax_softplus(4.0) / 4.0
    assert jnp.allclose(out, x * expected_scale)


def test_sst_scaling() -> None:
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = sst(x)
    n = 2
    t = jnp.tanh(x)
    s = jnp.matmul(t, t)
    expected = s * (float(n) ** -2)
    assert jnp.allclose(out, expected)


def test_sss_scaling() -> None:
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = sss(x)
    n = 2
    t = jax_sigmoid(x)
    s = jnp.matmul(t, t)
    expected = s * (float(n) ** -2)
    assert jnp.allclose(out, expected)
