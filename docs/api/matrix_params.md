# `matnets.MatrixParams`

`MatrixParams` stores the weights and bias for matrix primitives:

```python
from matnets import MatrixParams
```

Dense parameters use:

```text
W: (q, p, n, n)
B: (q, n, n)
```

`MatrixParams` is registered as a JAX pytree, so it works with `jax.jit`,
`jax.vmap`, `jax.grad`, and nested dictionaries/lists of parameters.
