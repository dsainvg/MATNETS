# Dense Primitive

The core operation is:

```python
jnp.einsum("qpak,pkc->qac", W, x) + B
```

Under the square-matrix contract:

```text
a == n
k == n
c == n
```

so the output is always `(q, n, n)`.
