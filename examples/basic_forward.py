import jax
import jax.numpy as jnp

import matnets as mtn

key = jax.random.key(0)
params = mtn.init(key, p=2, q=3, n=2)
x = jnp.arange(8.0).reshape(2, 2, 2) / 8.0

y = mtn.dense(params, x, activation=jax.nn.relu)

print("x:", x.shape)
print("y:", y.shape)
print(y)
