# Getting Started

## Install Locally

MATNETS targets Python 3.11+.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install pytest
```

MATNETS uses JAX. It does not depend on TensorFlow or PyTorch.

## First Dense Layer

```python
import jax
import jax.numpy as jnp

import matnets as mtn

params = mtn.init(jax.random.key(0), p=2, q=3, n=2)
x = jnp.ones((2, 2, 2))
y = mtn.dense(params, x, activation=jax.nn.relu)

print(y.shape)  # (3, 2, 2)
```

## Five Hidden Layers

The runnable class-style example is in `examples/five_hidden_net.py`.

```python
import jax
import jax.numpy as jnp

from examples.five_hidden_net import FiveHiddenNet

model = FiveHiddenNet(jax.random.key(42), input_neurons=3, n=2)
x = jnp.ones((3, 2, 2))
y = jax.jit(model.forward)(model.params, x)

print(y.shape)  # (1, 2, 2)
```

Run it:

```powershell
.\.venv\Scripts\python.exe examples\five_hidden_net.py
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
