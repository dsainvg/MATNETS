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
