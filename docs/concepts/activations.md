# Activations

Like pooling, activations in MATNETS can be element-wise (standard) or structural
(determinant-based).

### Element-wise Activations

Standard activations like `relu`, `leaky_relu`, and `elu` can be applied to
matrix-valued neurons. In this case, the scalar function is applied to every
entry in the $n \times n$ matrix independently.

### Determinant-based Matrix Activations (Gated and Scaled)

MATNETS introduces structural activations that treat the $n \times n$ neuron as
a single unit by gating or scaling based on its determinant.

- **`relud`**: Returns the input matrix if its determinant is positive,
  otherwise zeros it out. This ensures only orientation-preserving
  transformations pass.
- **`leaky_relud`**: Similar to `relud`, but scales the matrix by a small
  $\alpha$ if the determinant is non-positive, allowing some gradient flow.
- **`elu_powered`**: Returns the input matrix if the determinant is positive, else
  applies the matrix-exponential branch $\alpha(e^X - I)$. Note that the matrix
  exponential makes this operation relatively slow.
- **`elud`**: Scales the matrix by $\text{elu}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
  This keeps the scaling smooth and dimension-normalized.
- **`sigmoidd`**: Scales the matrix by $\text{sigmoid}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`tanhd`**: Scales the matrix by $\text{tanh}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
- **`softplusd`**: Scales the matrix by $\text{softplus}(\text{det}(X)^{1/n}) / \text{det}(X)^{1/n}$.
