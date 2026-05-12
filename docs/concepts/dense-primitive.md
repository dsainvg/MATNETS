# The Dense Primitive

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
