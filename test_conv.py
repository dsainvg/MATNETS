import jax
import jax.numpy as jnp
from matnets._params import MatrixParams
from matnets.lax.conv import matrix_conv2d, _same_padding

key = jax.random.PRNGKey(0)
q, p, kernel_y, kernel_x, n = 2, 3, 3, 3, 4
W = jax.random.normal(key, (q, p, kernel_y, kernel_x, n, n))
B = jax.random.normal(key, (q, n, n))
params = MatrixParams(W=W, B=B)

x = jax.random.normal(key, (10, 10, p, n, n))
out_orig = matrix_conv2d(params, x, stride=2, padding="SAME")

c = n
x_trans = jnp.transpose(x, (4, 0, 1, 2, 3))
x_reshaped = jnp.reshape(x_trans, (c, x.shape[0], x.shape[1], p * n))

W_trans = jnp.transpose(W, (0, 4, 1, 5, 2, 3))
W_reshaped = jnp.reshape(W_trans, (q * n, p * n, kernel_y, kernel_x))

pad_y = _same_padding(kernel_y)
pad_x = _same_padding(kernel_x)

out_new_conv = jax.lax.conv_general_dilated(
    x_reshaped, W_reshaped, 
    window_strides=(2, 2), 
    padding=(pad_y, pad_x), 
    dimension_numbers=("NHWC", "OIHW", "NHWC")
)
out_new_reshaped = jnp.reshape(out_new_conv, (c, out_new_conv.shape[1], out_new_conv.shape[2], q, n))
out_new = jnp.transpose(out_new_reshaped, (1, 2, 3, 4, 0)) + B

print(jnp.allclose(out_orig, out_new, atol=1e-5))
print('Shapes:', out_orig.shape, out_new.shape)
print('Max diff:', jnp.max(jnp.abs(out_orig - out_new)))
