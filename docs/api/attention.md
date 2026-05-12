# Attention

### `matnets.lax.matrix_attention`

Computes attention over a sequence of matrix-valued tokens.

```python
from matnets.lax import matrix_attention

out = matrix_attention(params, Q, K, V, score_fn=None)
```

**Arguments:**

- `params` (MatrixParams | None): Optional dense projection parameters applied to the final aggregated context vectors.
- `Q` (jax.Array): Query tokens, shape `(tokens_q, p, n, n)`.
- `K` (jax.Array): Key tokens, shape `(tokens_k, p, n, n)`.
- `V` (jax.Array): Value tokens, shape `(tokens_k, p, n, n)`.
- `score_fn` (Callable, optional): A function returning a scalar given a Query and Key matrix. Defaults to a scaled Frobenius inner product.
