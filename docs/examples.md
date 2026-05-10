# Examples

## Basic Forward Pass

```powershell
.\.venv\Scripts\python.exe examples\basic_forward.py
```

This creates one dense layer and applies it to an input shaped `(2, 2, 2)`.

## Five Hidden Layers

```powershell
.\.venv\Scripts\python.exe examples\five_hidden_net.py
```

This example defines a small class:

```python
model = FiveHiddenNet(key, input_neurons=3, hidden_neurons=4, n=2)
y = jax.jit(model.forward)(model.params, x)
```

The output shape is `(1, 2, 2)`.

## Architecture Walkthrough

```powershell
.\.venv\Scripts\python.exe examples\matrix_architectures.py
```

This file checks shape flow through:

- MLP
- batched MLP with `jax.vmap`
- gradients with `jax.grad`
- RNN with `jax.lax.scan`
- LSTM with `jax.lax.scan`
- Frobenius attention
- residual block

## Pooling and CNNs

```powershell
.\.venv\Scripts\python.exe examples\10_pooling.py
```

Demonstrates downsampling a 1D sequence using `maxd_pool1d` and `avgd_pool1d`.
Shows how pooling integrates into a standard Flax model with matrix
convolutions.

## Where Parallelization Happens

The dense operation:

```python
jnp.einsum("qpak,pkc->qac", W, x)
```

is the main kernel. JAX can compile it with `jit`, map it over batches or token
sequences with `vmap`, differentiate it with `grad`, and call it repeatedly
inside `lax.scan`.
