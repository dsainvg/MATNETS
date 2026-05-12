# Recurrent Networks

MATNETS recurrent cells are designed to be used natively with `jax.lax.scan`. They manage hidden states that are stacks of matrices.

### RNN

```python
from matnets.nn import rnn_step

carry, outputs = jax.lax.scan(
    lambda h, x_t: rnn_step(params, h, x_t),
    h0,
    sequence
)
```

### LSTM

```python
from matnets.nn import lstm_step

# params must contain specific keys: "i", "f", "g", "o"
carry, outputs = jax.lax.scan(
    lambda carry, x_t: lstm_step(params, carry, x_t),
    (h0, c0),
    sequence
)
```

### GRU

```python
from matnets.nn import gru_step

# params must contain specific keys: "z", "r", "n"
carry, outputs = jax.lax.scan(
    lambda h, x_t: gru_step(params, h, x_t),
    h0,
    sequence
)
```
