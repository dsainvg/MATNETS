# `matnets.lax.matrix_conv1d`

```python
from matnets.lax import matrix_conv1d

y = matrix_conv1d(params, x, stride=1, padding="VALID")
```

Expected shapes:

```text
params.W: (q, p, r, n, n)
params.B: (q, n, n)
x:        (t, p, n, n)
y:        (t_out, q, n, n)
```

`r` is the 1D kernel size.
