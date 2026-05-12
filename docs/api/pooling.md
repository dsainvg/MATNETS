# Pooling

Pooling downsamples spatial or sequential dimensions while preserving the matrix-neuron contract.

=== "Standard Element-wise Pooling"

    Operates identically to traditional pooling but applied independently to each $(i, j)$ matrix entry.

    - `max_pool1d` / `max_pool2d`: Spatial maximum.
    - `avg_pool1d` / `avg_pool2d`: Arithmetic mean.
    - `sum_pool1d` / `sum_pool2d`: Arithmetic sum.

=== "Determinant-based Structural Pooling"

    Operates on the entire matrix holistically based on its volume-preserving properties.

    - **`maxd_pool1d` / `maxd_pool2d`**: Selects the single matrix in the window with the highest determinant.
    - **`avgd_pool1d` / `avgd_pool2d`**: Computes $\sum \frac{1}{\det(M)} M$ for all matrices in the window.
