import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matnets as mtn
import optax


# Define a deep 7-layer Multi-Layer Perceptron architecture
class MLP7(nn.Module):
    @nn.compact
    def __call__(self, t):
        def dense_layer(name, x, p, q, act=None):
            w = self.param(f"{name}_W", nn.initializers.lecun_normal(), (q, p, 2, 2))
            b = self.param(f"{name}_B", nn.initializers.zeros, (q, 2, 2))
            return mtn.dense(mtn.MatrixParams(W=w, B=b), x, activation=act or (lambda z: z))

        t = dense_layer("h0", t, 2, 8, jax.nn.silu)
        for i in range(1, 7):
            t = dense_layer(f"h{i}", t, 8, 8, jax.nn.silu)
        out = dense_layer("out", t, 8, 2)
        return jnp.array([out[0].mean(), out[1].mean()])


model = MLP7()

# Generate complex synthetic binary classification data
key = jax.random.key(3)
key_x, key_noise = jax.random.split(key)
x = jax.random.normal(key_x, (64, 2, 2, 2))
y_idx = jnp.logical_xor(
    x[:, 0].sum(axis=(1, 2)) > 0,
    x[:, 1].sum(axis=(1, 2)) > 0,
).astype(jnp.int32)
flip = jax.random.bernoulli(key_noise, 0.1, shape=y_idx.shape)
y_idx = jnp.where(flip, 1 - y_idx, y_idx)
y = jax.nn.one_hot(y_idx, 2)

# Initialize Flax training state tracking model parameters and optimizer
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(4), x[0]),
    tx=optax.adam(8e-3),
)


# Define a compiled training step for optimizing model parameters
@jax.jit
def train_step(s):
    def loss_fn(p):
        logits = jax.vmap(lambda t: s.apply_fn(p, t))(x)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled training loop using JAX lax.fori_loop for performance
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 140, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("mlp_7hl_loss:", float(loss))
