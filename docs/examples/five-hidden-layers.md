# Five Hidden Layers

```powershell
.\.venv\Scripts\python.exe examples\five_hidden_net.py
```

This example defines a small class:

```python
model = FiveHiddenNet(key, input_neurons=3, hidden_neurons=4, n=2)
y = jax.jit(model.forward)(model.params, x)
```

The output shape is `(1, 2, 2)`.
