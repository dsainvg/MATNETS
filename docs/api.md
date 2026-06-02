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

## `matnets.activations`

```python
from matnets.activations import (
    relu, relud, leaky_relu, leaky_relud,
    elu, elu_powered, elud,
    sigmoid, sigmoidd,
    tanh, tanhd,
    softplus, softplusd
)
```

### Standard and Determinant-Gated/Scaled Activations

MATNETS supports standard element-wise activations and determinant-based structural activations (either gated by sign or scaled by $1/n$-th root determinant, where $n$ is the matrix dimension).

#### Element-Wise Activations (Standard)

- **`relu(x)`**: Standard element-wise ReLU.
- **`leaky_relu(x, negative_slope=0.01)`**: Standard element-wise leaky ReLU.
- **`elu(x, alpha=1.0)`**: Standard element-wise ELU.
- **`sigmoid(x)`**: Standard element-wise sigmoid.
- **`tanh(x)`**: Standard element-wise tanh.
- **`softplus(x)`**: Standard element-wise softplus.

#### Determinant-Gated Activations (Branching)

- **`relud(x)`**: Determinant-gated ReLU. Returns $X$ if $\text{det}(X) > 0$, else $0$.
- **`leaky_relud(x, negative_slope=0.01)`**: Determinant-gated leaky ReLU. Returns $X$ if $\text{det}(X) > 0$, else `negative_slope * X`.
- **`elu_powered(x, alpha=1.0)`**: Determinant-gated ELU (matrix exponential). Returns $X$ if $\text{det}(X) > 0$, else `alpha * (expm(X) - I)`.

#### Determinant-Scaled Activations (Smooth Scaling)

These functions scale the input matrix by $\text{fn}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$, using the $1/n$-th root of the determinant for dimension-normalized stability (with small-epsilon clamping on $\text{det}(X)$ for numerical safety):

- **`elud(x, alpha=1.0)`**: Scales by $\text{elu}(\text{det}(X)^{1/n}, \alpha) / \text{det}(X)^{1/n}$.
- **`sigmoidd(x)`**: Scales by $\text{sigmoid}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`tanhd(x)`**: Scales by $\text{tanh}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`softplusd(x)`**: Scales by $\text{softplus}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.

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
from matnets.activations import relud
y = matnets.dense(params, x, activation=relud)
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

- `maxd_pool1d/2d`: Selects the single matrix in the window with the highest determinant.
- `avgd_pool1d/2d`: Computes $\sum \frac{1}{\text{det}(M)^{1/n}} M$ for all matrices $M$ in the window, where $n$ is the matrix dimension.

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

### `embed_sequence`

```python
from matnets.utils import embed_sequence
import numpy as np

seq = np.array([1, 2, 3, 4, 5])  # 1D Sequence: (T,)
out = embed_sequence(seq, n=3, axis=0)
# Shape: (5, 3, 3)

# For multiple channels / batch:
seq_mc = np.zeros((10, 5)) # (T, C)
out_mc = embed_sequence(seq_mc, n=3, axis=0)
# Shape: (10, 5, 3, 3)
```

`embed_sequence` extracts a symmetric `n x n` time-history embedding over a given time axis, ideal for audio, time-series, or other sequentially streaming data. 

For every time step `t` along the target sequence `axis`, this backwardly extracts history up to `n` steps, constructing a symmetric matrix where distance from the diagonal corresponds naturally to the delay. Previous states prior to `t=0` are strictly zero-padded.

It supports native mapping over 1D, 2D, and 3D data formats (e.g. `(T,)`, `(N, T)`, `(N, T, C)`) without interfering with non-sequential dimensions.

