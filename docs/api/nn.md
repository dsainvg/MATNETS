# `matnets.nn`

`matnets.nn` contains recurrent wiring patterns built from `dense`.

```python
from matnets.nn import rnn_step, lstm_step, gru_step
```

These functions are intended to be used with `jax.lax.scan`.

### RNN

```python
carry, outputs = jax.lax.scan(
    lambda h, x_t: rnn_step(params, h, x_t),
    h0,
    sequence,
)
```

### LSTM

```python
from matnets.activations import sss, sst

carry, outputs = jax.lax.scan(
    lambda carry, x_t: lstm_step(
        params,
        carry,
        x_t,
        activations=(sss, sst), # Optional: defaults to (sigmoid, tanh)
    ),
    (h0, c0),
    sequence,
)
```

LSTM params must contain keys `"i"`, `"f"`, `"g"`, and `"o"`.

The `lstm_step` uses matrix multiplications (`jnp.matmul`) for the cell state and hidden state updates. You can pass a custom `activations` tuple to configure the gate and state activations (e.g. `(gate_act, state_act)`).

### GRU

```python
carry, outputs = jax.lax.scan(
    lambda h, x_t: gru_step(params, h, x_t),
    h0,
    sequence,
)
```

GRU params must contain keys `"z"`, `"r"`, and `"n"`.
