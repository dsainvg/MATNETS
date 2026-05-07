import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matnets as mtn
import optax


# Define an attention-based Transformer model analyzing input token dependencies
class TinyTransformer(nn.Module):
    @nn.compact
    def __call__(self, tokens):
        def dense_layer(name, x, p, q, act=None):
            w = self.param(f"{name}_W", nn.initializers.lecun_normal(), (q, p, 2, 2))
            b = self.param(f"{name}_B", nn.initializers.zeros, (q, 2, 2))
            return mtn.dense(mtn.MatrixParams(W=w, B=b), x, activation=act or (lambda z: z))

        q = jax.vmap(lambda t: dense_layer("q", t, 3, 3))(tokens)
        k = jax.vmap(lambda t: dense_layer("k", t, 3, 3))(tokens)
        v = jax.vmap(lambda t: dense_layer("v", t, 3, 3))(tokens)
        a = mtn.lax.matrix_attention(None, q, k, v)
        ff = jax.vmap(lambda t: dense_layer("ff1", t, 3, 6, jax.nn.gelu))(a)
        ff = jax.vmap(lambda t: dense_layer("ff2", t, 6, 3))(ff)
        out = dense_layer("out", ff.mean(axis=0), 3, 2)
        return jnp.array([out[0].mean(), out[1].mean()])


model = TinyTransformer()

# Initialize synthetic multidimensional token features corresponding to masked targets
key = jax.random.key(8)
key_x, key_noise = jax.random.split(key)
x = jax.random.normal(key_x, (40, 7, 3, 2, 2))
y_idx = jnp.logical_xor(
    x[:, :, 0].mean(axis=(1, 2, 3)) > 0,
    x[:, :, 2].mean(axis=(1, 2, 3)) > 0,
).astype(jnp.int32)
flip = jax.random.bernoulli(key_noise, 0.35, shape=y_idx.shape)
y_idx = jnp.where(flip, 1 - y_idx, y_idx)
y = jax.nn.one_hot(y_idx, 2)

# Instantiate optimization context associating the initial model state with Adam
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(9), x[0]),
    tx=optax.adam(8e-3),
)


# Define a compiled transformation encapsulating the loss gradient updates
@jax.jit
def train_step(s):
    def loss_fn(p):
        logits = jax.vmap(lambda tok: s.apply_fn(p, tok))(x)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled high-performance loop driving iterative optimization
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 55, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("transformer_loss:", float(loss))
