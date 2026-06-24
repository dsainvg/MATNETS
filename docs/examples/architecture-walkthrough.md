# Architecture Walkthrough

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
