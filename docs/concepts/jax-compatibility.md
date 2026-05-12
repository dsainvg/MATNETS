# JAX Transforms Compatibility

A key design principle of MATNETS is full compatibility with JAX's functional transformations. The dense einsum operation is the main parallel kernel.

| JAX Transform | MATNETS Usage |
| :--- | :--- |
| `jax.jit` | Compiles the `mtn.dense` or custom forward pass into highly optimized XLA code. |
| `jax.vmap` | Adds batch dimensions `(batch, p, n, n)` or token dimensions without rewriting the core equations. |
| `jax.grad` | Computes gradients through the matrix contractions for training. |
| `jax.lax.scan` | Efficiently loops over sequences for RNNs, managing the matrix-valued hidden states. |
