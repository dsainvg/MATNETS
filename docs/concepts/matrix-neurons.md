# Matrix-Neurons

A traditional dense layer usually maps vectors:

```text
x: (p)
W: (q, p)
y: (q)
```

MATNETS maps stacks of square matrices:

```text
x: (p, n, n)
W: (q, p, n, n)
B: (q, n, n)
y: (q, n, n)
```

`p` is the input neuron count. `q` is the output neuron count. `n` is the
matrix size inside each neuron.
