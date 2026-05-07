import jax
import jax.numpy as jnp

from examples.five_hidden_net import FiveHiddenNet


def test_five_hidden_net_class_forward_and_jit() -> None:
    keys = jax.random.split(jax.random.key(123), 2)
    model = FiveHiddenNet(keys[0], input_neurons=3, hidden_neurons=4, n=2)
    x = jax.random.normal(keys[1], (3, 2, 2))

    eager = model(x)
    compiled = model.jit()(model.params, x)

    assert eager.shape == (1, 2, 2)
    assert compiled.shape == (1, 2, 2)
    assert jnp.allclose(eager, compiled)
