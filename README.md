# MATNETS

MATNETS is a small JAX library for matrix-neuron neural network experiments.
Each neuron carries an `n x n` matrix instead of a scalar.

The user documentation lives in [`docs/`](docs/index.md):

- [`docs/index.md`](docs/index.md): overview
- [`docs/getting-started.md`](docs/getting-started.md): install and first model
- [`docs/concepts.md`](docs/concepts.md): matrix-neuron shapes and JAX transforms
- [`docs/api.md`](docs/api.md): API guide
- [`docs/examples.md`](docs/examples.md): runnable examples
- [`docs/development.md`](docs/development.md): local development commands

Quick check:

```powershell
.\.venv\Scripts\python.exe examples\five_hidden_net.py
.\.venv\Scripts\python.exe -m pytest
```

MIT license.
