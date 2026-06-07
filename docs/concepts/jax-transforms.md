# JAX Transforms

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
