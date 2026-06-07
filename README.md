# MATNETS

MATNETS is a small JAX library for matrix-neuron neural network experiments.
Each neuron carries an `n x n` matrix instead of a scalar.

The user documentation lives in [`docs/`](docs/index.md):

- [`docs/index.md`](docs/index.md): overview
- [`docs/getting-started/index.md`](docs/getting-started/index.md): install and first model
- [`docs/concepts/index.md`](docs/concepts/index.md): matrix-neuron shapes and JAX transforms
- [`docs/api/index.md`](docs/api/index.md): API guide
- [`docs/examples/index.md`](docs/examples/index.md): runnable examples
- [`docs/getting-started/development.md`](docs/getting-started/development.md): local development commands

Quick check:

```powershell
.\.venv\Scripts\python.exe examples\five_hidden_net.py
.\.venv\Scripts\python.exe -m pytest
```

MIT license.
