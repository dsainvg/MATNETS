import jax
import jax.numpy as jnp
import pytest

from matnets import MatrixParams, dense, init
from matnets.lax import frobenius_score, matrix_attention, matrix_conv1d, matrix_conv2d
from matnets.nn import gru_step, lstm_step, rnn_step


def test_init_returns_dense_params_pytree() -> None:
    params = init(jax.random.key(0), p=2, q=3, n=4)

    assert params.W.shape == (3, 2, 4, 4)
    assert params.B.shape == (3, 4, 4)

    leaves = jax.tree.leaves(params)
    assert [leaf.shape for leaf in leaves] == [(3, 2, 4, 4), (3, 4, 4)]


def test_init_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="must all be positive"):
        init(jax.random.key(0), p=0, q=1, n=2)


def test_dense_supports_jit_vmap_and_grad() -> None:
    params = MatrixParams(
        W=jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
        B=jnp.array([[[1.0, 10.0], [2.0, 20.0]]]),
    )
    x = jnp.ones((1, 2, 2))

    output = jax.jit(dense)(params, x)

    assert output.shape == (1, 2, 2)
    assert jnp.allclose(output[0], jnp.array([[4.0, 13.0], [9.0, 27.0]]))

    batch = jnp.stack([x, x * 2.0])
    batched = jax.vmap(lambda item: dense(params, item))(batch)
    assert batched.shape == (2, 1, 2, 2)

    grad_params = jax.grad(lambda p: dense(p, x).sum())(params)
    assert grad_params.W.shape == params.W.shape
    assert grad_params.B.shape == params.B.shape


def test_dense_requires_square_weights_input_and_bias() -> None:
    x = jnp.ones((1, 2, 2))

    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 3)),
        B=jnp.zeros((1, 2, 2)),
    )
    with pytest.raises(ValueError, match="last two axes must match"):
        dense(params, x)

    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2)),
        B=jnp.zeros((1, 2, 2)),
    )
    with pytest.raises(ValueError, match="dense expects input shaped"):
        dense(params, jnp.ones((1, 2, 3)))

    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2)),
        B=jnp.zeros((1, 2, 1)),
    )
    with pytest.raises(ValueError, match="dense bias must be shaped"):
        dense(params, x)


def test_matrix_conv1d_slides_kernel() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]])

    output = matrix_conv1d(params, x)

    assert output.shape == (2, 1, 1, 1)
    assert output[:, 0, 0, 0].tolist() == pytest.approx([3.0, 5.0])


def test_matrix_conv1d_same_padding_keeps_sequence_length() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 3, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]])

    output = matrix_conv1d(params, x, padding="SAME")

    assert output.shape == (3, 1, 1, 1)
    assert output[:, 0, 0, 0].tolist() == pytest.approx([3.0, 6.0, 5.0])


def test_matrix_conv1d_requires_square_matrix_weights() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 3)),
        B=jnp.zeros((1, 2, 2)),
    )
    x = jnp.ones((3, 1, 3, 3))

    with pytest.raises(ValueError, match="last two axes must match"):
        matrix_conv1d(params, x)


def test_matrix_conv1d_requires_matching_input_and_bias_axes() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 2)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.ones((3, 1, 2, 2))

    with pytest.raises(ValueError, match="bias must be shaped"):
        matrix_conv1d(params, x)

    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 2)),
        B=jnp.zeros((1, 2, 2)),
    )
    x = jnp.ones((3, 1, 3, 3))

    with pytest.raises(ValueError, match="input axes must match"):
        matrix_conv1d(params, x)


def test_matrix_conv1d_rejects_bad_stride_and_padding() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.ones((3, 1, 1, 1))

    with pytest.raises(ValueError, match="stride must be positive"):
        matrix_conv1d(params, x, stride=0)
    with pytest.raises(ValueError, match="padding must be"):
        matrix_conv1d(params, x, padding="FULL")  # type: ignore[arg-type]


def test_matrix_conv2d_slides_kernel() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.array(
        [
            [[[[1.0]]], [[[2.0]]]],
            [[[[3.0]]], [[[4.0]]]],
        ]
    )

    output = matrix_conv2d(params, x)

    assert output.shape == (1, 1, 1, 1, 1)
    assert output[0, 0, 0, 0, 0] == pytest.approx(10.0)


def test_matrix_conv2d_same_padding_keeps_grid_shape() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 3, 3, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.ones((2, 2, 1, 1, 1))

    output = matrix_conv2d(params, x, padding="SAME")

    assert output.shape == (2, 2, 1, 1, 1)
    assert jnp.allclose(output[:, :, 0, 0, 0], jnp.full((2, 2), 4.0))


def test_matrix_conv2d_requires_square_matrix_weights() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 2, 3)),
        B=jnp.zeros((1, 2, 2)),
    )
    x = jnp.ones((2, 2, 1, 3, 3))

    with pytest.raises(ValueError, match="last two axes must match"):
        matrix_conv2d(params, x)


def test_matrix_conv2d_requires_matching_input_and_bias_axes() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 2, 2)),
        B=jnp.zeros((1, 1, 1)),
    )
    x = jnp.ones((2, 2, 1, 2, 2))

    with pytest.raises(ValueError, match="bias must be shaped"):
        matrix_conv2d(params, x)

    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2, 2, 2)),
        B=jnp.zeros((1, 2, 2)),
    )
    x = jnp.ones((2, 2, 1, 3, 3))

    with pytest.raises(ValueError, match="input axes must match"):
        matrix_conv2d(params, x)


def test_matrix_attention_uses_pluggable_scores() -> None:
    Q = jnp.ones((2, 1, 1, 1))
    K = jnp.ones((2, 1, 1, 1))
    V = jnp.array([[[[2.0]]], [[[4.0]]]])

    output = matrix_attention(None, Q, K, V, score_fn=lambda q, k: jnp.array(0.0))

    assert output.shape == (2, 1, 1, 1)
    assert output[:, 0, 0, 0].tolist() == pytest.approx([3.0, 3.0])


def test_frobenius_score_matches_elementwise_sum() -> None:
    Q = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    K = jnp.array([[[2.0, 3.0], [4.0, 5.0]]])

    assert frobenius_score(Q, K) == pytest.approx(40.0)


def test_matrix_attention_can_project_context_tokens() -> None:
    Q = jnp.ones((2, 1, 2, 2))
    K = jnp.ones((2, 1, 2, 2))
    V = jnp.ones((2, 1, 2, 2))
    params = MatrixParams(
        W=jnp.ones((1, 1, 2, 2)),
        B=jnp.zeros((1, 2, 2)),
    )

    output = matrix_attention(params, Q, K, V, score_fn=lambda q, k: jnp.array(0.0))

    assert output.shape == (2, 1, 2, 2)
    assert jnp.allclose(output, jnp.full((2, 1, 2, 2), 2.0))


def test_matrix_attention_rejects_non_square_or_mismatched_tokens() -> None:
    Q = jnp.ones((2, 1, 2, 3))
    K = jnp.ones((2, 1, 2, 3))
    V = jnp.ones((2, 1, 2, 3))

    with pytest.raises(ValueError, match="square matrix tokens"):
        matrix_attention(None, Q, K, V)

    Q = jnp.ones((2, 1, 2, 2))
    K = jnp.ones((2, 2, 2, 2))
    V = jnp.ones((2, 1, 2, 2))

    with pytest.raises(ValueError, match="must share neuron and matrix axes"):
        matrix_attention(None, Q, K, V)


def test_rnn_step_composes_dense() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 2, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    carry = jnp.array([[[1.0]]])
    x = jnp.array([[[2.0]]])

    next_carry, output = rnn_step(params, carry, x, activation=lambda value: value)

    assert next_carry.shape == (1, 1, 1)
    assert output[0, 0, 0] == pytest.approx(3.0)


def test_rnn_step_works_inside_scan() -> None:
    params = MatrixParams(
        W=jnp.ones((1, 2, 1, 1)),
        B=jnp.zeros((1, 1, 1)),
    )
    h0 = jnp.zeros((1, 1, 1))
    seq = jnp.ones((3, 1, 1, 1))

    final, outputs = jax.lax.scan(
        lambda carry, x_t: rnn_step(params, carry, x_t, activation=lambda value: value),
        h0,
        seq,
    )

    assert final.shape == (1, 1, 1)
    assert outputs.shape == (3, 1, 1, 1)
    assert outputs[:, 0, 0, 0].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_rnn_step_with_sst() -> None:
    from matnets.activations import sst
    params = MatrixParams(
        W=jnp.ones((1, 2, 2, 2)),
        B=jnp.zeros((1, 2, 2)),
    )
    carry = jnp.ones((1, 2, 2))
    x = jnp.ones((1, 2, 2))

    next_carry, output = rnn_step(params, carry, x, activation=sst)

    assert next_carry.shape == (1, 2, 2)
    assert output.shape == (1, 2, 2)


def test_rnn_step_with_sss() -> None:
    from matnets.activations import sss
    params = MatrixParams(
        W=jnp.ones((1, 2, 2, 2)),
        B=jnp.zeros((1, 2, 2)),
    )
    carry = jnp.ones((1, 2, 2))
    x = jnp.ones((1, 2, 2))

    next_carry, output = rnn_step(params, carry, x, activation=sss)

    assert next_carry.shape == (1, 2, 2)
    assert output.shape == (1, 2, 2)


def test_lstm_step_returns_hidden_and_cell_matrices() -> None:
    params = {
        gate: MatrixParams(
            W=jnp.ones((1, 2, 1, 1)),
            B=jnp.zeros((1, 1, 1)),
        )
        for gate in ("i", "f", "g", "o")
    }
    h0 = jnp.zeros((1, 1, 1))
    c0 = jnp.zeros((1, 1, 1))
    x = jnp.ones((1, 1, 1))

    (next_h, next_c), output = lstm_step(params, (h0, c0), x)

    assert next_h.shape == (1, 1, 1)
    assert next_c.shape == (1, 1, 1)
    assert output.shape == (1, 1, 1)


def test_lstm_step_with_custom_activations() -> None:
    from matnets.activations import sss, sst
    params = {
        gate: MatrixParams(
            W=jnp.ones((1, 2, 2, 2)),
            B=jnp.zeros((1, 2, 2)),
        )
        for gate in ("i", "f", "g", "o")
    }
    h0 = jnp.ones((1, 2, 2))
    c0 = jnp.ones((1, 2, 2))
    x = jnp.ones((1, 2, 2))

    (next_h, next_c), output = lstm_step(params, (h0, c0), x, activations=(sss, sst))

    assert next_h.shape == (1, 2, 2)
    assert next_c.shape == (1, 2, 2)
    assert output.shape == (1, 2, 2)


def test_gru_step_returns_matrix_state() -> None:
    params = {
        gate: MatrixParams(
            W=jnp.ones((1, 2, 1, 1)),
            B=jnp.zeros((1, 1, 1)),
        )
        for gate in ("z", "r", "n")
    }
    carry = jnp.zeros((1, 1, 1))
    x = jnp.ones((1, 1, 1))

    next_carry, output = gru_step(params, carry, x)

    assert next_carry.shape == (1, 1, 1)
    assert output.shape == (1, 1, 1)
