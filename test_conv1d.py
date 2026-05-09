import jax
import jax.numpy as jnp
from matnets._params import MatrixParams
from matnets.lax.conv import matrix_conv1d, _same_padding

key = jax.random.PRNGKey(0)
q, p, kernel, n = 2, 3, 3, 4
W = jax.random.normal(key, (q, p, kernel, n, n))
B = jax.random.normal(key, (q, n, n))
params = MatrixParams(W=W, B=B)

x = jax.random.normal(key, (10, p, n, n))
out_orig = matrix_conv1d(params, x, stride=2, padding="SAME")

c = n
# x: (t, p, k, c) -> transpose to (c, t, p, k) -> reshape to (c, t, p*k)
x_trans = jnp.transpose(x, (3, 0, 1, 2))
x_reshaped = jnp.reshape(x_trans, (c, x.shape[0], p * n))

# W: (q, p, r, a, k) -> transpose to (q, a, p, k, r) -> reshape to (q*a, p*k, r)
W_trans = jnp.transpose(W, (0, 3, 1, 4, 2))
W_reshaped = jnp.reshape(W_trans, (q * n, p * n, kernel))

pad_l, pad_r = _same_padding(kernel)

out_new_conv = jax.lax.conv_general_dilated(
    x_reshaped, W_reshaped, 
    window_strides=(2,), 
    padding=((pad_l, pad_r),), 
    dimension_numbers=("NWC", "OIW", "NWC")
)
# out_new_conv: (c, out_t, q*a)
out_new_reshaped = jnp.reshape(out_new_conv, (c, out_new_conv.shape[1], q, n))
# -> transpose to (out_t, q, a, c)
out_new = jnp.transpose(out_new_reshaped, (1, 2, 3, 0)) + B

print(jnp.allclose(out_orig, out_new, atol=1e-5))
print('Shapes:', out_orig.shape, out_new.shape)
print('Max diff:', jnp.max(jnp.abs(out_orig - out_new)))
