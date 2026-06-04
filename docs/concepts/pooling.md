# Pooling

Pooling in MATNETS can be element-wise (standard) or structural
(determinant-based).

### Structural Pooling

Instead of comparing every scalar entry, structural pooling looks at the matrix
as a whole. `maxd_pool` selects the matrix with the highest determinant from a
window, preserving the structural integrity of the selected "winning" neuron
activation. `avgd_pool` weights each matrix contribution by its inverse
$1/n$-th root determinant: $\sum \frac{1}{\text{det}(M)^{1/n}} M$, where $n$ is the matrix dimension.
