# MATNETS Examples

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe examples\basic_forward.py
.\.venv\Scripts\python.exe examples\matrix_architectures.py
.\.venv\Scripts\python.exe examples\five_hidden_net.py
```

## Where Parallelization Happens

The biggest parallel work is inside `mtn.dense()`:

```python
jnp.einsum("qpak,pkc->qac", W, x)
```

JAX lowers that contraction to compiled array kernels. Every architecture in
`matrix_architectures.py` mostly gets its speed by calling this primitive many
times under JAX transforms.

Parallelization points in the examples:

- `jax.jit(mlp_forward)`: compiles the full MLP into a staged function.
- `jax.vmap(mlp_forward, in_axes=(None, 0))`: runs the same MLP over a batch.
- `jax.grad(...)`: differentiates through the same dense contractions.
- `jax.lax.scan(...)`: keeps RNN/LSTM time recurrence explicit while each step's
  dense gate computations stay compiled and vectorized.
- `jax.vmap(...)` in attention: projects every token with the same dense op.
- nested `jax.vmap(...)` in `matrix_attention`: computes all query/key pair
  scores without writing Python loops over tokens.

So the main pattern is:

```text
write one square-matrix primitive -> compose architectures -> apply JAX transforms
```
