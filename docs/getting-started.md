# Getting Started

Welcome to MATNETS! This guide will help you install the library and write your first matrix-neuron neural network.

MATNETS is built on top of [JAX](https://github.com/google/jax), providing a fast, differentiable framework for matrix-based operations.

---

## 1. Installation

MATNETS requires **Python 3.11+**. We recommend setting up a virtual environment before installing.

=== "Windows (PowerShell)"

    ```powershell
    # 1. Create a virtual environment
    python -m venv .venv

    # 2. Activate the environment (optional but recommended)
    .\.venv\Scripts\Activate.ps1

    # 3. Install MATNETS in editable mode
    python -m pip install -e .

    # 4. Install test dependencies (optional)
    python -m pip install pytest
    ```

=== "Linux / macOS"

    ```bash
    # 1. Create a virtual environment
    python3 -m venv .venv

    # 2. Activate the environment
    source .venv/bin/activate

    # 3. Install MATNETS in editable mode
    python -m pip install -e .

    # 4. Install test dependencies (optional)
    python -m pip install pytest
    ```

!!! info "No TensorFlow or PyTorch Required"
    MATNETS strictly uses JAX as its backend for array operations and automatic differentiation. It does not depend on TensorFlow or PyTorch.

---

## 2. Your First Dense Layer

Let's write a simple script to understand how a basic dense layer works in MATNETS.

```python title="first_layer.py"
import jax
import jax.numpy as jnp
import matnets as mtn

# 1. Initialize parameters
# p=2 (input neurons), q=3 (output neurons), n=2 (matrix dimension)
key = jax.random.key(0)
params = mtn.init(key, p=2, q=3, n=2)

# 2. Create input data
# Shape: (p, n, n) -> 2 matrix-neurons, each 2x2
x = jnp.ones((2, 2, 2))

# 3. Apply the dense layer
# We use the determinant-based relud activation
y = mtn.dense(params, x, activation=mtn.activations.relud)

# 4. Check the output shape
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}") # Expected: (3, 2, 2)
```

Run this script:

```bash
python first_layer.py
```

---

## 3. Building a Multi-Layer Network

MATNETS functions are pure JAX functions. You can compose them easily to build deeper networks. Here is an example of a network with five hidden layers, demonstrating how to wrap MATNETS calls in a class structure.

We will use the provided `examples/five_hidden_net.py`.

```python title="Building a Deeper Network"
import jax
import jax.numpy as jnp

# Import the class-based model from our examples
from examples.five_hidden_net import FiveHiddenNet

# Initialize the model
# input_neurons=3, internal matrices are 2x2
model = FiveHiddenNet(jax.random.key(42), input_neurons=3, n=2)

# Create some dummy input
x = jnp.ones((3, 2, 2))

# JIT-compile the forward pass for performance
# This is a standard JAX pattern!
fast_forward = jax.jit(model.forward)
y = fast_forward(model.params, x)

print(f"Deep network output shape: {y.shape}") # Expected: (1, 2, 2)
```

To run the full example directly from the repository:

```bash
python examples/five_hidden_net.py
```

---

## 4. Running the Test Suite

To ensure everything is installed and functioning correctly on your machine, run the test suite:

```bash
python -m pytest
```

If all tests pass, you are ready to start experimenting with matrix-valued neural networks!

## What's Next?

- Read the [**Concepts**](concepts/matrix-neurons.md) page to understand the math and JAX integration.
- Check the [**API Guide**](api/core-layers.md) for full module documentation.
