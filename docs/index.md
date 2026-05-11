---
hide:
  - navigation
  - toc
---

<div class="hero" markdown="1">
  <h1>MATNETS</h1>
  <p class="description">A JAX library for matrix-neuron neural network experiments.</p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![JAX](https://img.shields.io/badge/powered%20by-JAX-informational.svg)](https://github.com/google/jax)

  <br><br>

  <a href="getting-started/" class="button button--primary">Get Started</a>
  <a href="concepts/" class="button button--secondary">Read Concepts</a>
</div>

<div class="feature-grid">
<div class="feature-card" markdown="1">

### Matrix-Valued Neurons
In traditional neural networks, neurons carry scalars. In MATNETS, each neuron carries an `n x n` matrix, preserving rich geometric information.

</div>
<div class="feature-card" markdown="1">

### Determinant Activations
Use structural pooling and determinant-gated activations to filter operations based on matrix orientation and volume properties.

</div>
<div class="feature-card" markdown="1">

### JAX Optimized
Fully compatible with JAX transforms like `jax.jit`, `jax.vmap`, and `jax.grad`. Fast, composable, and XLA-accelerated.

</div>
</div>

---

## Core Shape Contract

MATNETS operations are built on a consistent square-matrix shape contract. The core dense primitive takes an input of $p$ matrix-neurons and produces $q$ output matrix-neurons, each containing an $n \times n$ matrix:

```python
params.W: (q, p, n, n)
params.B: (q, n, n)
x:        (p, n, n)
output:   (q, n, n)
```

## Quick Start Overview

Create and run your first matrix-neuron layer in just a few lines:

```python
import jax
import jax.numpy as jnp
import matnets as mtn

# Initialize parameters for a layer: 2 inputs -> 3 outputs, 2x2 matrices
params = mtn.init(jax.random.key(0), p=2, q=3, n=2)

# Create some dummy input (2 neurons, 2x2 matrices)
x = jnp.ones((2, 2, 2))

# Apply the dense layer with a determinant-based ReLU activation
y = mtn.dense(params, x, activation=mtn.activations.relud)

print(y.shape) # Output: (3, 2, 2)
```

## Read Next

- [**Getting Started**](getting-started.md): Detailed installation instructions and introductory tutorials.
- [**Concepts**](concepts.md): Deep dive into matrix-neuron mathematics, shapes, and JAX compatibility.
- [**API Guide**](api.md): Comprehensive reference for all public functions and classes.
- [**Examples**](examples.md): Explore runnable example scripts demonstrating different architectures.
- [**Development**](development.md): Guide for contributing, running tests, and local development.
