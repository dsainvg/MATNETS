# API Guide

## `matnets.MatrixParams`

`MatrixParams` stores the weights and bias for matrix primitives:

```python
from matnets import MatrixParams
```

Dense parameters use:

```text
W: (q, p, n, n)
B: (q, n, n)
```

`MatrixParams` is registered as a JAX pytree, so it works with `jax.jit`,
`jax.vmap`, `jax.grad`, and nested dictionaries/lists of parameters.

## `matnets.init`

```python
params = matnets.init(key, p=2, q=3, n=4)
```

Creates:

```text
params.W: (3, 2, 4, 4)
params.B: (3, 4, 4)
```

Weights use Glorot-uniform initialization. Bias starts at zero.

## `matnets.dense`

```python
y = matnets.dense(params, x)
```

Expected shapes:

```text
params.W: (q, p, n, n)
params.B: (q, n, n)
x:        (p, n, n)
y:        (q, n, n)
```

With activation:

```python
y = matnets.dense(params, x, activation=jax.nn.relu)
```

The core operation is:

```python
jnp.einsum("qpak,pkc->qac", params.W, x) + params.B
```

## `matnets.lax.matrix_conv1d`

```python
from matnets.lax import matrix_conv1d

y = matrix_conv1d(params, x, stride=1, padding="VALID")
```

Expected shapes:

```text
params.W: (q, p, r, n, n)
params.B: (q, n, n)
x:        (t, p, n, n)
y:        (t_out, q, n, n)
```

`r` is the 1D kernel size.

## `matnets.lax.matrix_conv2d`

```python
from matnets.lax import matrix_conv2d

y = matrix_conv2d(params, x, stride=(1, 1), padding="SAME")
```

Expected shapes:

```text
params.W: (q, p, h, w, n, n)
params.B: (q, n, n)
x:        (height, width, p, n, n)
y:        (height_out, width_out, q, n, n)
```

## `matnets.conv`

Pooling primitives for downsampling sequential and grid data.

```python
from matnets.conv import max_pool1d, maxd_pool1d, avg_pool1d, avgd_pool1d
from matnets.conv import max_pool2d, maxd_pool2d, avg_pool2d, avgd_pool2d
```

Standard pooling (`max_pool`, `avg_pool`, `sum_pool`) operates on matrix
elements. Determinant-based pooling selects or weights matrices based on their
determinant.

### Standard Pooling

- `max_pool1d/2d`: Element-wise maximum within the window.
- `avg_pool1d/2d`: Standard arithmetic mean of matrices in the window.
- `sum_pool1d/2d`: Standard sum of matrices in the window.

### Determinant Pooling

- `maxd_pool1d/2d`: Selects the single matrix in the window with the highest
  determinant.
- `avgd_pool1d/2d`: Computes $\sum \frac{1}{\text{det}(M)} M$ for all matrices
  $M$ in the window.

Expected 1D shapes:

```text
x: (t, p, n, n) or (batch, t, p, n, n)
y: (t_out, p, n, n) or (batch, t_out, p, n, n)
```

Expected 2D shapes:

```text
x: (y, x, p, n, n) or (batch, y, x, p, n, n)
y: (y_out, x_out, p, n, n) or (batch, y_out, x_out, p, n, n)
```

## `matnets.lax.matrix_attention`

```python
from matnets.lax import matrix_attention

out = matrix_attention(None, Q, K, V)
```

Expected token shapes:

```text
Q:   (tokens_q, p, n, n)
K:   (tokens_k, p, n, n)
V:   (tokens_k, p, n, n)
out: (tokens_q, p, n, n)
```

By default the score is a scaled Frobenius inner product. You can pass a custom
`score_fn` that receives one query token and one key token and returns a scalar.

If `params` is not `None`, each aggregated output token is projected through
`matnets.dense(params, token)`.

## `matnets.nn`

`matnets.nn` contains recurrent wiring patterns built from `dense`.

```python
from matnets.nn import rnn_step, lstm_step, gru_step
```

These functions are intended to be used with `jax.lax.scan`.

### RNN

```python
carry, outputs = jax.lax.scan(
    lambda h, x_t: rnn_step(params, h, x_t),
    h0,
    sequence,
)
```

### LSTM

```python
carry, outputs = jax.lax.scan(
    lambda carry, x_t: lstm_step(params, carry, x_t),
    (h0, c0),
    sequence,
)
```

LSTM params must contain keys `"i"`, `"f"`, `"g"`, and `"o"`.

### GRU

```python
carry, outputs = jax.lax.scan(
    lambda h, x_t: gru_step(params, h, x_t),
    h0,
    sequence,
)
```

GRU params must contain keys `"z"`, `"r"`, and `"n"`.

## `matnets.utils`

Data preprocessing utilities for MATNETS.

```python
from matnets.utils import embed_pixels
import numpy as np

imgs = np.zeros((2, 10, 10, 3))  # (Batch, H, W, Channels)
windows = embed_pixels(imgs, n=3, spatial_axes=(1, 2), interleave=False)
# Shape: (2, 10, 10, 3, 3, 3)
```

`embed_pixels` extracts an `n x n` (or `n` for 1D) local neighborhood around
each element. The function automatically applies zero padding so the output
spatial dimensions match the input spatial dimensions, with the new window
dimensions appended to the end of the shape.

If `interleave=True` (or a tuple of booleans per axis), the order of elements
along the spatial axes is permuted according to an interleaved block pattern.
