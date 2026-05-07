import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matnets as mtn
import optax


# Define a single-neuron linear model using MATNETS dense primitive
class SingleNeuronLinear(nn.Module):
    n: int = 2

    @nn.compact
    def __call__(self, x):
        w = self.param("W", nn.initializers.lecun_normal(), (1, 1, self.n, self.n))
        b = self.param("B", nn.initializers.zeros, (1, self.n, self.n))
        return mtn.dense(mtn.MatrixParams(W=w, B=b), x)


model = SingleNeuronLinear()

# Generate synthetic training data with added noise
key_x, key_noise = jax.random.split(jax.random.key(0))
x = jax.random.normal(key_x, (32, 1, 2, 2))
y = 2.0 * x + 0.3 + 0.25 * jax.random.normal(key_noise, x.shape)

# Initialize Flax training state with Adam optimizer
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(jax.random.key(1), x[0]),
    tx=optax.adam(3e-2),
)


# Define a compiled training step computing loss and gradients
@jax.jit
def train_step(s):
    def loss_fn(p):
        pred = jax.vmap(lambda t: s.apply_fn(p, t))(x)
        return jnp.mean(optax.l2_loss(predictions=pred, targets=y))

    loss, grads = jax.value_and_grad(loss_fn)(s.params)
    return s.apply_gradients(grads=grads), loss


# Define a compiled training loop using JAX lax.fori_loop for performance
@jax.jit
def train_epochs(s):
    def body(_, carry):
        next_state, _ = train_step(carry)
        return next_state

    return jax.lax.fori_loop(0, 90, body, s)


state = train_epochs(state)
loss = train_step(state)[1]

print("linear_regression_loss:", float(loss))
