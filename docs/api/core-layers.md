# Core Layers

### `matnets.dense`

The fundamental matrix-neuron layer, computing $\mathbf{Y} = \mathbf{W}\mathbf{X} + \mathbf{B}$ via tensor contraction.

```python
y = mtn.dense(params, x, activation=None)
```

**Arguments:**

- `params` (MatrixParams): The weights and biases for the layer.
- `x` (jax.Array): The input stack of matrix-neurons. Shape must be `(p, n, n)`.
- `activation` (Callable, optional): An activation function to apply to the output.

**Returns:**

- `jax.Array`: The output stack of matrix-neurons. Shape is `(q, n, n)`.
