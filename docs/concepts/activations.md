# Activations

Activations in MATNETS can be categorized into two types: standard element-wise functions and determinant-based gated functions.

### Element-wise Activations

Standard activations like `relu`, `leaky_relu`, and `elu` can be applied directly. The scalar function $f(z)$ is applied to every entry $x_{ij}$ in the matrix independently:

$$
\mathbf{Y}_{ij} = f(\mathbf{X}_{ij})
$$

### Determinant-based Matrix Activations (Gated)

To fully leverage the matrix structure, MATNETS provides activations that gate the entire matrix based on its determinant, ensuring the layer only passes transformations that meet specific geometric criteria (e.g., orientation preservation).

- **`relud`**: Determinant-gated ReLU. It passes the matrix unchanged if its determinant is positive, otherwise it zeros out the entire matrix.

$$
  \text{relud}(\mathbf{X}) = \begin{cases}
  \mathbf{X} & \text{if } \det(\mathbf{X}) > 0 \\
  \mathbf{0} & \text{otherwise}
  \end{cases}
$$

- **`leaky_relud`**: Similar to `relud`, but scales the matrix by a small scalar $\alpha$ when the determinant is non-positive, preventing dead neurons while still heavily penalizing orientation-reversing transformations.

$$
  \text{leaky\_relud}(\mathbf{X}) = \begin{cases}
  \mathbf{X} & \text{if } \det(\mathbf{X}) > 0 \\
  \alpha \mathbf{X} & \text{otherwise}
  \end{cases}
$$

- **`elud`**: A continuous alternative that applies the matrix exponential branch when the determinant is non-positive.

$$
  \text{elud}(\mathbf{X}) = \begin{cases}
  \mathbf{X} & \text{if } \det(\mathbf{X}) > 0 \\
  \alpha (\exp(\mathbf{X}) - \mathbf{I}) & \text{otherwise}
  \end{cases}
$$
