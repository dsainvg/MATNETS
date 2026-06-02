# Concepts

## Matrix-Neurons

A traditional dense layer usually maps vectors:

```text
x: (p)
W: (q, p)
y: (q)
```

MATNETS maps stacks of square matrices:

```text
x: (p, n, n)
W: (q, p, n, n)
B: (q, n, n)
y: (q, n, n)
```

`p` is the input neuron count. `q` is the output neuron count. `n` is the
matrix size inside each neuron.

## Dense Primitive

The core operation is:

```python
jnp.einsum("qpak,pkc->qac", W, x) + B
```

Under the square-matrix contract:

```text
a == n
k == n
c == n
```

so the output is always `(q, n, n)`.

## Bias

The bias is a full matrix:

```text
B: (q, n, n)
```

Each output matrix-neuron gets its own complete matrix bias.

## JAX Transforms

MATNETS functions are ordinary JAX functions. You can transform them with:

```python
jax.jit(forward)
jax.vmap(forward, in_axes=(None, 0))
jax.grad(loss)
jax.lax.scan(step, carry, sequence)
```

The main parallel work is the dense einsum. `vmap` adds batch or token axes
around it. `scan` handles recurrence over time while each step still uses
compiled dense contractions.

## Recurrent State

RNN, LSTM, and GRU hidden states are stacks of matrices:

```text
H: (hidden_neurons, n, n)
C: (hidden_neurons, n, n)
```

Gates are also matrix-valued, so an LSTM forget gate has one value per matrix
entry, not just one scalar per neuron.

## Pooling

Pooling in MATNETS can be element-wise (standard) or structural
(determinant-based).

### Structural Pooling

Instead of comparing every scalar entry, structural pooling looks at the matrix
as a whole. `maxd_pool` selects the matrix with the highest determinant from a
window, preserving the structural integrity of the selected "winning" neuron
activation. `avgd_pool` weights each matrix contribution by its inverse
$1/n$-th root determinant: $\sum \frac{1}{\text{det}(M)^{1/n}} M$, where $n$ is the matrix dimension.

## Activations

Like pooling, activations in MATNETS can be element-wise (standard) or structural
(determinant-based).

### Element-wise Activations

Standard activations like `relu`, `leaky_relu`, and `elu` can be applied to
matrix-valued neurons. In this case, the scalar function is applied to every
entry in the $n \times n$ matrix independently.

### Determinant-based Matrix Activations (Gated and Scaled)

MATNETS introduces structural activations that treat the $n \times n$ neuron as
a single unit by gating or scaling based on its determinant.

- **`relud`**: Returns the input matrix if its determinant is positive,
  otherwise zeros it out. This ensures only orientation-preserving
  transformations pass.
- **`leaky_relud`**: Similar to `relud`, but scales the matrix by a small
  $\alpha$ if the determinant is non-positive, allowing some gradient flow.
- **`elu_powered`**: Returns the input matrix if the determinant is positive, else
  applies the matrix-exponential branch $\alpha(e^X - I)$. Note that the matrix
  exponential makes this operation relatively slow.
- **`elud`**: Scales the matrix by $\text{elu}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
  This keeps the scaling smooth and dimension-normalized.
- **`sigmoidd`**: Scales the matrix by $\text{sigmoid}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`tanhd`**: Scales the matrix by $\text{tanh}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`softplusd`**: Scales the matrix by $\text{softplus}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
