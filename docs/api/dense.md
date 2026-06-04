# `matnets.dense`

```python
y = matnets.dense(params, x)
```

Expected shapes:

```text
params.W: (q, p, n, n)
params.B: (q, n, n)
x:        (p, n, n)
y:        (q, n, n)
```

With activation:

```python
from matnets.activations import relud
y = matnets.dense(params, x, activation=relud)
```

The core operation is:

```python
jnp.einsum("qpak,pkc->qac", params.W, x) + params.B
```
