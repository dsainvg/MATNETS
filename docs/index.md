# MATNETS

MATNETS is a JAX library for matrix-neuron neural network experiments.

In a traditional neural network, a neuron carries one scalar activation. In
MATNETS, each neuron carries an `n x n` matrix. A layer maps a stack of input
matrix-neurons to a stack of output matrix-neurons.

## Core Shape Contract

The core dense primitive uses:

```text
params.W: (q, p, n, n)
params.B: (q, n, n)
x:        (p, n, n)
output:   (q, n, n)
```

`p` is the input neuron count. `q` is the output neuron count. `n` is the matrix
size for every neuron.

## Read Next

- [Getting Started](getting-started/index.md): install MATNETS and run a first dense layer.
- [Concepts](concepts/index.md): understand matrix-neuron shapes and JAX transforms.
- [API Guide](api/index.md): see each public function and its expected shapes.
- [Examples](examples/index.md): run the included examples.
- [Development](getting-started/development.md): run tests and local checks.
