# Recurrent State

RNN, LSTM, and GRU hidden states are stacks of matrices:

```text
H: (hidden_neurons, n, n)
C: (hidden_neurons, n, n)
```

Gates are also matrix-valued, so an LSTM forget gate has one value per matrix
entry, not just one scalar per neuron.
