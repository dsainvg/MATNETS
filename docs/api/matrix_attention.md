# `matnets.lax.matrix_attention`

```python
from matnets.lax import matrix_attention

out = matrix_attention(None, Q, K, V)
```

Expected token shapes:

```text
Q:   (tokens_q, p, n, n)
K:   (tokens_k, p, n, n)
V:   (tokens_k, p, n, n)
out: (tokens_q, p, n, n)
```

By default the score is a scaled Frobenius inner product. You can pass a custom
`score_fn` that receives one query token and one key token and returns a scalar.

If `params` is not `None`, each aggregated output token is projected through
`matnets.dense(params, token)`.
