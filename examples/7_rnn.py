import flax.linen as nnx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matnets as mtn
from matnets import nn
import optax


# Define a foundational Recurrent Neural Network aggregating input sequence state
class RNNNet(nnx.Module):
    @nnx.compact
    def __call__(self, seq):
        cell = mtn.MatrixParams(
            W=self.param("cell_W", nnx.initializers.lecun_normal(), (6, 8, 2, 2)),
            B=self.param("cell_B", nnx.initializers.zeros, (6, 2, 2)),
        )
        outp = mtn.MatrixParams(
            W=self.param("out_W", nnx.initializers.lecun_normal(), (2, 6, 2, 2)),
            B=self.param("out_B", nnx.initializers.zeros, (2, 2, 2)),
        )
        h0 = jnp.zeros((6, 2, 2))
        _, hs = jax.lax.scan(
            lambda h, x_t: nn.rnn_step(cell, h, x_t, activation=jax.nn.tanh), h0, seq
        )
        out = mtn.dense(outp, hs[-1], activation=jax.nn.gelu)
        return jnp.array([out[0].mean(), out[1].mean()])


model = RNNNet()

# Produce synthesized sequential observations testing temporal dependencies
key = jax.random.key(7)
key_x, key_noise = jax.random.split(key)
x = jax.random.normal(key_x, (48, 10, 2, 2, 2))
y_idx = jnp.logical_xor(
    x[:, :, 0].mean(axis=(1, 2, 3)) > 0,
    x[:, -1, 1].mean(axis=(1, 2)) > 0,
).astype(jnp.int32)
flip = jax.random.bernoulli(key_noise, 0.35, shape=y_idx.shape)
y_idx = jnp.where(flip, 1 - y_idx, y_idx)
y = jax.nn.one_hot(y_idx, 2)

# Set up parameter tracking alongside the optimization scheme
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(8), x[0]),
    tx=optax.adam(1e-2),
)


# Define a compiled stochastic gradient descent progression over the dataset
@jax.jit
def train_step(s):
    def loss_fn(p):
        logits = jax.vmap(lambda seq: s.apply_fn(p, seq))(x)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled loop executing repeated forward and backward passes
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 55, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("rnn_loss:", float(loss))
