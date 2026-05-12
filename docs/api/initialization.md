# Initialization & Parameters

### `MatrixParams`

`MatrixParams` is a custom JAX PyTree used to store the weights and biases for matrix primitives.

```python
from matnets import MatrixParams
```

It stores tensors with the following required shapes for a dense layer:

- **`W`**: `(q, p, n, n)` - Weight matrices mapping `p` inputs to `q` outputs.
- **`B`**: `(q, n, n)` - Bias matrices for each of the `q` outputs.

Because it is registered as a PyTree, `MatrixParams` works natively with `jax.jit`, `jax.vmap`, `jax.grad`, and tree-map utilities.

---

### `matnets.init`

Initializes parameters for a dense layer using Glorot-uniform initialization for weights and zeros for biases.

```python
import matnets as mtn
params = mtn.init(key, p=2, q=3, n=4)
```

**Arguments:**

- `key` (jax.random.PRNGKey): The random seed.
- `p` (int): Number of input matrix-neurons.
- `q` (int): Number of output matrix-neurons.
- `n` (int): Dimension of the square matrices.

**Returns:**

- `MatrixParams`: A PyTree containing initialized `W` and `B`.
