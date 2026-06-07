# `matnets.activations`

```python
from matnets.activations import (
    relu, relud, leaky_relu, leaky_relud,
    elu, elu_powered, elud,
    sigmoid, sigmoidd,
    tanh, tanhd,
    softplus, softplusd
)
```

### Standard and Determinant-Gated/Scaled Activations

MATNETS supports standard element-wise activations and determinant-based structural activations (either gated by sign or scaled by $1/n$-th root determinant, where $n$ is the matrix dimension).

#### Element-Wise Activations (Standard)

- **`relu(x)`**: Standard element-wise ReLU.
- **`leaky_relu(x, negative_slope=0.01)`**: Standard element-wise leaky ReLU.
- **`elu(x, alpha=1.0)`**: Standard element-wise ELU.
- **`sigmoid(x)`**: Standard element-wise sigmoid.
- **`tanh(x)`**: Standard element-wise tanh.
- **`softplus(x)`**: Standard element-wise softplus.
- **`sss(x)`**: Scaled squared sigmoid. Applies element-wise sigmoid, squares the resulting matrices, and scales them by $n^{-1}$.
- **`sst(x)`**: Scaled squared tanh. Applies element-wise tanh, squares the resulting matrices, and scales them by $n^{-1}$.

#### Determinant-Gated Activations (Branching)

- **`relud(x)`**: Determinant-gated ReLU. Returns $X$ if $\text{det}(X) > 0$, else $0$.
- **`leaky_relud(x, negative_slope=0.01)`**: Determinant-gated leaky ReLU. Returns $X$ if $\text{det}(X) > 0$, else `negative_slope * X`.
- **`elu_powered(x, alpha=1.0)`**: Determinant-gated ELU (matrix exponential). Returns $X$ if $\text{det}(X) > 0$, else `alpha * (expm(X) - I)`.

#### Determinant-Scaled Activations (Smooth Scaling)

These functions scale the input matrix by $\text{fn}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$, using the $1/n$-th root of the determinant for dimension-normalized stability (with small-epsilon clamping on $\text{det}(X)$ for numerical safety):

- **`elud(x, alpha=1.0)`**: Scales by $\text{elu}(\text{det}(X)^{1/n}, \alpha) / \text{det}(X)^{1/n}$.
- **`sigmoidd(x)`**: Scales by $\text{sigmoid}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`tanhd(x)`**: Scales by $\text{tanh}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`softplusd(x)`**: Scales by $\text{softplus}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
