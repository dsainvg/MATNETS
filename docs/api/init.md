# `matnets.init`

```python
params = matnets.init(key, p=2, q=3, n=4)
```

Creates:

```text
params.W: (3, 2, 4, 4)
params.B: (3, 4, 4)
```

Weights use Glorot-uniform initialization. Bias starts at zero.
