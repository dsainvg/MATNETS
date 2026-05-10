import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import matnets as mtn
from matnets.conv import avgd_pool1d, maxd_pool1d


# Define a 1D Convolutional Neural Network with various pooling layers
class CNN1DPooling(nn.Module):
    @nn.compact
    def __call__(self, seq):
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, 2, 3, 2, 2)),
            B=self.param("c1_B", nn.initializers.zeros, (4, 2, 2)),
        )
        c2 = mtn.MatrixParams(
            W=self.param("c2_W", nn.initializers.lecun_normal(), (4, 4, 3, 2, 2)),
            B=self.param("c2_B", nn.initializers.zeros, (4, 2, 2)),
        )
        outp = mtn.MatrixParams(
            W=self.param("out_W", nn.initializers.lecun_normal(), (2, 4, 2, 2)),
            B=self.param("out_B", nn.initializers.zeros, (2, 2, 2)),
        )

        # 1. Apply Convolution
        h = jax.nn.gelu(mtn.lax.matrix_conv1d(c1, seq, padding="SAME"))

        # 2. Apply a Max Determinant Pool
        h = maxd_pool1d(h, window_size=2, stride=2, padding="VALID")

        # 3. Apply Convolution
        h = jax.nn.gelu(mtn.lax.matrix_conv1d(c2, h, padding="SAME"))

        # 4. Apply Average Determinant Pool
        h = avgd_pool1d(h, window_size=2, stride=2, padding="VALID")

        out = mtn.dense(outp, h.mean(axis=0))
        return jnp.array([out[0].mean(), out[1].mean()])


model = CNN1DPooling()

# Construct synthetic sequential data for a sequence classification task
key = jax.random.key(4)
key_x, key_noise = jax.random.split(key)
x = jax.random.normal(key_x, (48, 12, 2, 2, 2))  # (batch, seq_len, p, n, n)
y_idx = jnp.logical_xor(
    x[:, :, 0].mean(axis=(1, 2, 3)) > 0,
    x[:, :, 1].mean(axis=(1, 2, 3)) > 0,
).astype(jnp.int32)
flip = jax.random.bernoulli(key_noise, 0.35, shape=y_idx.shape)
y_idx = jnp.where(flip, 1 - y_idx, y_idx)
y = jax.nn.one_hot(y_idx, 2)

# Set up optimizer and training state binding parameters
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(5), x[0]),
    tx=optax.adam(1e-2),
)


# Define a compiled optimization step calculating cross-entropy loss
@jax.jit
def train_step(s):
    def loss_fn(p):
        logits = jax.vmap(lambda seq: s.apply_fn(p, seq))(x)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled loop executing training steps efficiently
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 55, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("cnn1d_pooling_loss:", float(loss))
