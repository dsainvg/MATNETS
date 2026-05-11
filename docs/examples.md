# Examples

The `examples/` directory in the repository contains several runnable scripts demonstrating how to build and train models with MATNETS.

---

## 1. Basic Forward Pass

Demonstrates the absolute minimum code required to initialize and run a single matrix-neuron layer.

```bash
python examples/basic_forward.py
```

**Key Takeaways:**

- Initializes parameters using `mtn.init()`.
- Constructs a dummy input of shape `(2, 2, 2)`.
- Runs `mtn.dense()` and verifies the output shape.

---

## 2. Five Hidden Layers

Shows how to encapsulate MATNETS operations inside a Python class, similar to how one might structure a PyTorch `nn.Module` or Flax `nn.Module`.

```bash
python examples/five_hidden_net.py
```

**What it demonstrates:**

- Iterating through multiple layers, updating the feature map `x`.
- Managing a list of `MatrixParams` objects.
- JIT-compiling the entire class method `jax.jit(model.forward)`.

---

## 3. Architecture Walkthrough

A comprehensive testbed script that verifies shape propagation through various advanced architectural patterns.

```bash
python examples/matrix_architectures.py
```

**Architectures tested:**

- Standard multi-layer perceptrons (MLP).
- Batched execution using `jax.vmap`.
- Gradient computation using `jax.grad`.
- Recurrent sequence processing with `jax.lax.scan` (both simple RNN and LSTM).
- Scaled Frobenius Matrix Attention.
- Residual (skip) connections.

---

## 4. Pooling and Convolutions

Demonstrates how to process sequential (or spatial) data using structural pooling.

```bash
python examples/10_pooling.py
```

**What it demonstrates:**

- Downsampling sequences using `maxd_pool1d` (picking the highest determinant) and `avgd_pool1d` (inverse-determinant weighted sum).
- Integrating MATNETS convolutions (`matrix_conv1d`) with pooling inside a standard sequential model.

---

## 5. Structural Activations

Contrasts standard element-wise activations with MATNETS' unique determinant-gated activations.

```bash
python examples/11_activations.py
```

**What it demonstrates:**

- Applying `relu` (element-wise) vs. `relud` (determinant-gated).
- Applying `elu` (element-wise) vs. `elud` (determinant-gated matrix exponential).
- How gating preserves or filters entire matrix-neurons based on orientation.

---

## Integrating with JAX Ecosystem

MATNETS relies heavily on standard JAX functions for parallelization and optimization.

```python title="JAX Integration Examples"
# 1. Compilation
# JIT compile the dense kernel for speed
fast_dense = jax.jit(mtn.dense)

# 2. Batching
# Map the dense operation over a batch dimension (axis 0 of inputs)
batched_dense = jax.vmap(mtn.dense, in_axes=(None, 0))

# 3. Gradients
# Compute gradients of a loss function with respect to MatrixParams
grads = jax.grad(loss_fn)(params, x, y_true)

# 4. Recurrence
# Loop an RNN step over a time sequence
carry, outputs = jax.lax.scan(rnn_step, initial_state, sequence)
```
