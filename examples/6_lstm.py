import flax.linen as nnx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matnets as mtn
from matnets import nn
import optax


# Define a recurrent network architecture mapping sequences using LSTM cells
class LSTMNet(nnx.Module):
    @nnx.compact
    def __call__(self, seq):
        def gate(name):
            return mtn.MatrixParams(
                W=self.param(f"{name}_W", nnx.initializers.lecun_normal(), (6, 8, 2, 2)),
                B=self.param(f"{name}_B", nnx.initializers.zeros, (6, 2, 2)),
            )

        cell = {"i": gate("i"), "f": gate("f"), "g": gate("g"), "o": gate("o")}
        outp = mtn.MatrixParams(
            W=self.param("out_W", nnx.initializers.lecun_normal(), (2, 6, 2, 2)),
            B=self.param("out_B", nnx.initializers.zeros, (2, 2, 2)),
        )
        h0 = jnp.zeros((6, 2, 2))
        c0 = jnp.zeros((6, 2, 2))
        (_, _), hs = jax.lax.scan(lambda c, x_t: nn.lstm_step(cell, c, x_t), (h0, c0), seq)
        out = mtn.dense(outp, hs[-1], activation=jax.nn.tanh)
        return jnp.array([out[0].mean(), out[1].mean()])


model = LSTMNet()

# Build synthetic sequence data correlating temporal endpoints
key = jax.random.key(6)
x = jax.random.normal(key, (40, 9, 2, 2, 2))
y_idx = (x[:, -1, 0].sum(axis=(1, 2)) > x[:, 0, 1].sum(axis=(1, 2))).astype(jnp.int32)
y = jax.nn.one_hot(y_idx, 2)

# Create train state binding model parameters and the Adam optimizer
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(7), x[0]),
    tx=optax.adam(8e-3),
)


# Define a compiled update step retrieving loss and propagating gradients
@jax.jit
def train_step(s):
    def loss_fn(p):
        logits = jax.vmap(lambda seq: s.apply_fn(p, seq))(x)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled continuous loop progressing the train state
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 260, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("lstm_loss:", float(loss))
