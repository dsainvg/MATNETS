# `matnets.conv`

Pooling primitives for downsampling sequential and grid data.

```python
from matnets.conv import max_pool1d, maxd_pool1d, avg_pool1d, avgd_pool1d
from matnets.conv import max_pool2d, maxd_pool2d, avg_pool2d, avgd_pool2d
```

Standard pooling (`max_pool`, `avg_pool`, `sum_pool`) operates on matrix
elements. Determinant-based pooling selects or weights matrices based on their
determinant.

### Standard Pooling

- `max_pool1d/2d`: Element-wise maximum within the window.
- `avg_pool1d/2d`: Standard arithmetic mean of matrices in the window.
- `sum_pool1d/2d`: Standard sum of matrices in the window.

### Determinant Pooling

- `maxd_pool1d/2d`: Selects the single matrix in the window with the highest determinant.
- `avgd_pool1d/2d`: Computes $\sum \frac{1}{\text{det}(M)^{1/n}} M$ for all matrices $M$ in the window, where $n$ is the matrix dimension.

Expected 1D shapes:

```text
x: (t, p, n, n) or (batch, t, p, n, n)
y: (t_out, p, n, n) or (batch, t_out, p, n, n)
```

Expected 2D shapes:

```text
x: (y, x, p, n, n) or (batch, y, x, p, n, n)
y: (y_out, x_out, p, n, n) or (batch, y_out, x_out, p, n, n)
```
