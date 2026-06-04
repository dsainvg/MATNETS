# Where Parallelization Happens

The dense operation:

```python
jnp.einsum("qpak,pkc->qac", W, x)
```

is the main kernel. JAX can compile it with `jit`, map it over batches or token
sequences with `vmap`, differentiate it with `grad`, and call it repeatedly
inside `lax.scan`.
