import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import matnets as mtn


# Define a 2D Convolutional Neural Network architecture for spatial data
class CNN2D(nn.Module):
    @nn.compact
    def __call__(self, img):
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, 2, 3, 3, 2, 2)),
            B=self.param("c1_B", nn.initializers.zeros, (4, 2, 2)),
        )
        c2 = mtn.MatrixParams(
            W=self.param("c2_W", nn.initializers.lecun_normal(), (4, 4, 3, 3, 2, 2)),
            B=self.param("c2_B", nn.initializers.zeros, (4, 2, 2)),
        )
        outp = mtn.MatrixParams(
            W=self.param("out_W", nn.initializers.lecun_normal(), (2, 4, 2, 2)),
            B=self.param("out_B", nn.initializers.zeros, (2, 2, 2)),
        )
        h = jax.nn.relu(mtn.lax.matrix_conv2d(c1, img, padding="SAME"))
        h = jax.nn.relu(mtn.lax.matrix_conv2d(c2, h, padding="SAME"))
        out = mtn.dense(outp, h.mean(axis=(0, 1)))
        return jnp.array([out[0].mean(), out[1].mean()])


model = CNN2D()

# Synthesize 2D structured image data mapping features to binary targets
key = jax.random.key(5)
key_x, key_noise = jax.random.split(key)
x = jax.random.normal(key_x, (40, 8, 8, 2, 2, 2))
y_idx = (
    x[:, :4, :4, 0].mean(axis=(1, 2, 3, 4)) > x[:, 4:, 4:, 1].mean(axis=(1, 2, 3, 4))
).astype(jnp.int32)
flip = jax.random.bernoulli(key_noise, 0.25, shape=y_idx.shape)
y_idx = jnp.where(flip, 1 - y_idx, y_idx)
y = jax.nn.one_hot(y_idx, 2)

# Establish the training environment via Flax TrainState
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(6), x[0]),
    tx=optax.adam(1e-2),
)


# Define a compiled JAX transformation computing loss gradients and state updates
@jax.jit
def train_step(s):
    def loss_fn(p):
        logits = jax.vmap(lambda im: s.apply_fn(p, im))(x)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled iterative training procedure over epochs
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 70, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("cnn2d_loss:", float(loss))
