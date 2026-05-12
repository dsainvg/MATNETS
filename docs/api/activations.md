# Activations

```python
from matnets.activations import (
    relu, relud, leaky_relu, leaky_relud, elu, elud
)
```

### Element-wise Activations

These apply a standard scalar function to every element of the matrix independently:

- **`relu(x)`**: Standard element-wise maximum with zero.
- **`leaky_relu(x, negative_slope=0.01)`**: Standard element-wise leaky ReLU.
- **`elu(x, alpha=1.0)`**: Standard element-wise ELU.

### Determinant-Gated Activations

These treat the $n \times n$ matrix as a single geometric unit and gate based on $\det(\mathbf{X})$.

- **`relud(x)`**: Returns $X$ if $\det(X) > 0$, else $0$.
- **`leaky_relud(x, negative_slope=0.01)`**: Returns $X$ if $\det(X) > 0$, else $\alpha \cdot X$.
- **`elud(x, alpha=1.0)`**: Returns $X$ if $\det(X) > 0$, else $\alpha(\exp(X) - I)$.
