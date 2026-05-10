# MATNETS Examples

Run from the repository root with the local virtualenv:

```powershell
.\.venv\Scripts\python.exe examples\1_linear_regression.py
.\.venv\Scripts\python.exe examples\2_mlp_3_hidden_layers.py
.\.venv\Scripts\python.exe examples\3_mlp_7_hidden_layers.py
.\.venv\Scripts\python.exe examples\4_cnn1d.py
.\.venv\Scripts\python.exe examples\5_cnn2d.py
.\.venv\Scripts\python.exe examples\6_lstm.py
.\.venv\Scripts\python.exe examples\7_rnn.py
.\.venv\Scripts\python.exe examples\8_transformer.py
.\.venv\Scripts\python.exe examples\10_pooling.py
```

Each script is intentionally short and includes:
- activation functions (`relu`, `gelu`, `silu`, `tanh`)
- built-in losses from `optax` (`l2_loss`, `softmax_cross_entropy`)
- optimizer updates via `optax.adam` in Flax `TrainState`
