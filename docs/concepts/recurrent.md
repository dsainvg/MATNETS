# Recurrent Architectures

In MATNETS, hidden states for Recurrent Neural Networks (RNNs), LSTMs, and GRUs are not vectors of scalars, but stacks of matrices.

```text
H: (hidden_neurons, n, n)
C: (hidden_neurons, n, n)
```

Consequently, the gates in an LSTM (input, forget, output, etc.) are also matrix-valued. This means a forget gate has an independent scalar value for every entry in the $n \times n$ matrix, allowing the network to selectively forget specific components of the learned transformation, rather than scaling the entire neuron uniformly.
