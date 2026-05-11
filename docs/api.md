# API Guide

This guide provides a comprehensive reference for the public MATNETS API, categorized by module functionality.

---

## Initialization & Parameters

### `MatrixParams`

`MatrixParams` is a custom JAX PyTree used to store the weights and biases for matrix primitives.

```python
from matnets import MatrixParams
```

It stores tensors with the following required shapes for a dense layer:

- **`W`**: `(q, p, n, n)` - Weight matrices mapping `p` inputs to `q` outputs.
- **`B`**: `(q, n, n)` - Bias matrices for each of the `q` outputs.

Because it is registered as a PyTree, `MatrixParams` works natively with `jax.jit`, `jax.vmap`, `jax.grad`, and tree-map utilities.

---

### `matnets.init`

Initializes parameters for a dense layer using Glorot-uniform initialization for weights and zeros for biases.

```python
import matnets as mtn
params = mtn.init(key, p=2, q=3, n=4)
```

**Arguments:**

- `key` (jax.random.PRNGKey): The random seed.
- `p` (int): Number of input matrix-neurons.
- `q` (int): Number of output matrix-neurons.
- `n` (int): Dimension of the square matrices.

**Returns:**

- `MatrixParams`: A PyTree containing initialized `W` and `B`.

---

## Core Layers

### `matnets.dense`

The fundamental matrix-neuron layer, computing $\mathbf{Y} = \mathbf{W}\mathbf{X} + \mathbf{B}$ via tensor contraction.

```python
y = mtn.dense(params, x, activation=None)
```

**Arguments:**

- `params` (MatrixParams): The weights and biases for the layer.
- `x` (jax.Array): The input stack of matrix-neurons. Shape must be `(p, n, n)`.
- `activation` (Callable, optional): An activation function to apply to the output.

**Returns:**

- `jax.Array`: The output stack of matrix-neurons. Shape is `(q, n, n)`.

---

## Convolutions

MATNETS extends matrix-neurons to spatial data via matrix-based convolutions.

=== "1D Convolution"

    ```python
    from matnets.lax import matrix_conv1d

    y = matrix_conv1d(params, x, stride=1, padding="VALID")
    ```

    **Expected Shapes:**

    - `params.W`: `(q, p, kernel_size, n, n)`
    - `params.B`: `(q, n, n)`
    - `x`: `(seq_len, p, n, n)`
    - `y`: `(out_seq_len, q, n, n)`

=== "2D Convolution"

    ```python
    from matnets.lax import matrix_conv2d

    y = matrix_conv2d(params, x, stride=(1, 1), padding="SAME")
    ```

    **Expected Shapes:**

    - `params.W`: `(q, p, height, width, n, n)`
    - `params.B`: `(q, n, n)`
    - `x`: `(h_in, w_in, p, n, n)`
    - `y`: `(h_out, w_out, q, n, n)`

---

## Activations

```python
from matnets.activations import (
    relu, relud, leaky_relu, leaky_relud, elu, elud
)
```

### Element-wise Activations

These apply a standard scalar function to every element of the matrix independently:

- **`relu(x)`**: Standard element-wise maximum with zero.
- **`leaky_relu(x, negative_slope=0.01)`**: Standard element-wise leaky ReLU.
- **`elu(x, alpha=1.0)`**: Standard element-wise ELU.

### Determinant-Gated Activations

These treat the $n \times n$ matrix as a single geometric unit and gate based on $\det(\mathbf{X})$.

- **`relud(x)`**: Returns $X$ if $\det(X) > 0$, else $0$.
- **`leaky_relud(x, negative_slope=0.01)`**: Returns $X$ if $\det(X) > 0$, else $\alpha \cdot X$.
- **`elud(x, alpha=1.0)`**: Returns $X$ if $\det(X) > 0$, else $\alpha(\exp(X) - I)$.

---

## Pooling

Pooling downsamples spatial or sequential dimensions while preserving the matrix-neuron contract.

=== "Standard Element-wise Pooling"

    Operates identically to traditional pooling but applied independently to each $(i, j)$ matrix entry.

    - `max_pool1d` / `max_pool2d`: Spatial maximum.
    - `avg_pool1d` / `avg_pool2d`: Arithmetic mean.
    - `sum_pool1d` / `sum_pool2d`: Arithmetic sum.

=== "Determinant-based Structural Pooling"

    Operates on the entire matrix holistically based on its volume-preserving properties.

    - **`maxd_pool1d` / `maxd_pool2d`**: Selects the single matrix in the window with the highest determinant.
    - **`avgd_pool1d` / `avgd_pool2d`**: Computes $\sum \frac{1}{\det(M)} M$ for all matrices in the window.

---

## Attention

### `matnets.lax.matrix_attention`

Computes attention over a sequence of matrix-valued tokens.

```python
from matnets.lax import matrix_attention

out = matrix_attention(params, Q, K, V, score_fn=None)
```

**Arguments:**

- `params` (MatrixParams | None): Optional dense projection parameters applied to the final aggregated context vectors.
- `Q` (jax.Array): Query tokens, shape `(tokens_q, p, n, n)`.
- `K` (jax.Array): Key tokens, shape `(tokens_k, p, n, n)`.
- `V` (jax.Array): Value tokens, shape `(tokens_k, p, n, n)`.
- `score_fn` (Callable, optional): A function returning a scalar given a Query and Key matrix. Defaults to a scaled Frobenius inner product.

---

## Recurrent Networks

MATNETS recurrent cells are designed to be used natively with `jax.lax.scan`. They manage hidden states that are stacks of matrices.

### RNN

```python
from matnets.nn import rnn_step

carry, outputs = jax.lax.scan(
    lambda h, x_t: rnn_step(params, h, x_t),
    h0,
    sequence
)
```

### LSTM

```python
from matnets.nn import lstm_step

# params must contain specific keys: "i", "f", "g", "o"
carry, outputs = jax.lax.scan(
    lambda carry, x_t: lstm_step(params, carry, x_t),
    (h0, c0),
    sequence
)
```

### GRU

```python
from matnets.nn import gru_step

# params must contain specific keys: "z", "r", "n"
carry, outputs = jax.lax.scan(
    lambda h, x_t: gru_step(params, h, x_t),
    h0,
    sequence
)
```

---

## Utilities

### `matnets.utils.embed_pixels`

A preprocessing utility to convert standard image tensors into overlapping matrix-valued neighborhoods.

```python
from matnets.utils import embed_pixels
import numpy as np

# A standard image batch: (Batch, H, W, Channels)
imgs = np.zeros((2, 10, 10, 3))

# Extract 3x3 local neighborhoods
windows = embed_pixels(imgs, n=3, spatial_axes=(1, 2), interleave=False)

# Shape: (2, 10, 10, 3, 3, 3)
# -> (Batch, H, W, Channels, n, n)
```

**Arguments:**

- `imgs` (np.ndarray | jax.Array): The input image tensor.
- `n` (int): The window size (will become the $n \times n$ matrix dimensions).
- `spatial_axes` (tuple): The axes corresponding to height and width.
- `interleave` (bool | tuple): If true, permutes the order of elements along the spatial axes.