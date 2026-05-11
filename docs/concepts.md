# Concepts

This guide details the mathematical and structural concepts that differentiate MATNETS from standard neural network libraries.

---

## 1. Matrix-Neurons

In a traditional dense layer, inputs and outputs are vectors of scalars, and weights form a 2D matrix. The transformation is defined as $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

```text title="Traditional Neural Network Shapes"
x: (p)          # p input scalars
W: (q, p)       # mapping p inputs to q outputs
y: (q)          # q output scalars
```

In MATNETS, the fundamental unit of activation is an $n \times n$ matrix. A layer maps a stack of $p$ input matrix-neurons to a stack of $q$ output matrix-neurons.

```text title="MATNETS Shapes"
x: (p, n, n)    # p input matrix-neurons
W: (q, p, n, n) # mapping p inputs to q outputs
B: (q, n, n)    # bias is a full matrix per output neuron
y: (q, n, n)    # q output matrix-neurons
```

Where:

- $p$: The number of input neurons.
- $q$: The number of output neurons.
- $n$: The spatial dimension of the square matrix carried by each neuron.

---

## 2. The Dense Primitive

The core operation of a MATNETS dense layer is not a simple matrix multiplication, but a tensor contraction defined by the following Einstein summation:

```python
jnp.einsum("qpak,pkc->qac", W, x) + B
```

Mathematically, for the $j$-th output neuron ($1 \leq j \leq q$), the output matrix $\mathbf{Y}_j \in \mathbb{R}^{n \times n}$ is computed as:

$$
\mathbf{Y}_j = \sum_{i=1}^{p} \mathbf{W}_{ji} \mathbf{X}_i + \mathbf{B}_j
$$

Where:

- $\mathbf{X}_i \in \mathbb{R}^{n \times n}$ is the $i$-th input matrix-neuron.
- $\mathbf{W}_{ji} \in \mathbb{R}^{n \times n}$ is the weight matrix connecting input $i$ to output $j$.
- $\mathbf{B}_j \in \mathbb{R}^{n \times n}$ is the complete matrix bias for output neuron $j$.

Under the square-matrix contract, the inner dimensions contract cleanly:

```text
a == n
k == n
c == n
```

Ensuring the output is always shaped `(q, n, n)`.

---

## 3. Structural Pooling

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

---

## 4. Activations

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

---

## 5. Recurrent Architectures

In MATNETS, hidden states for Recurrent Neural Networks (RNNs), LSTMs, and GRUs are not vectors of scalars, but stacks of matrices.

```text
H: (hidden_neurons, n, n)
C: (hidden_neurons, n, n)
```

Consequently, the gates in an LSTM (input, forget, output, etc.) are also matrix-valued. This means a forget gate has an independent scalar value for every entry in the $n \times n$ matrix, allowing the network to selectively forget specific components of the learned transformation, rather than scaling the entire neuron uniformly.

---

## 6. JAX Transforms Compatibility

A key design principle of MATNETS is full compatibility with JAX's functional transformations. The dense einsum operation is the main parallel kernel.

| JAX Transform | MATNETS Usage |
| :--- | :--- |
| `jax.jit` | Compiles the `mtn.dense` or custom forward pass into highly optimized XLA code. |
| `jax.vmap` | Adds batch dimensions `(batch, p, n, n)` or token dimensions without rewriting the core equations. |
| `jax.grad` | Computes gradients through the matrix contractions for training. |
| `jax.lax.scan` | Efficiently loops over sequences for RNNs, managing the matrix-valued hidden states. |
