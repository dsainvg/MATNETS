# `matnets.lax.matrix_conv2d`

```python
from matnets.lax import matrix_conv2d

y = matrix_conv2d(params, x, stride=(1, 1), padding="SAME")
```

Expected shapes:

```text
params.W: (q, p, h, w, n, n)
params.B: (q, n, n)
x:        (height, width, p, n, n)
y:        (height_out, width_out, q, n, n)
```
