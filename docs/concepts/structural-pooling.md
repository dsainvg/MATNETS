# Structural Pooling

Standard pooling operations (like max-pooling or average-pooling) typically operate on scalar elements independently. MATNETS introduces **structural pooling**, which treats the entire $n \times n$ matrix as a single entity, often using the matrix determinant to evaluate its magnitude or volume-preserving properties.

### Determinant Max Pooling (`maxd_pool`)

Instead of finding the element-wise maximum across a spatial window, `maxd_pool` evaluates the determinant of every matrix in the window and selects the single matrix with the highest determinant. This preserves the internal structure (and thus the geometric transformation) of the "winning" neuron.

$$
\mathbf{Y} = \mathbf{X}_{k^*} \quad \text{where} \quad k^* = \arg\max_{k \in \text{window}} \det(\mathbf{X}_k)
$$

### Determinant Average Pooling (`avgd_pool`)

`avgd_pool` computes a weighted sum of the matrices in the window, where the weights are derived from the inverse determinant, prioritizing matrices that represent transformations with smaller volume expansion.

$$
\mathbf{Y} = \sum_{k \in \text{window}} \frac{1}{|\det(\mathbf{X}_k)| + \epsilon} \mathbf{X}_k
$$

*(Note: Exact implementations may vary slightly to handle numerical stability and zero determinants).*
