# Matrix-Neurons

In a traditional dense layer, inputs and outputs are vectors of scalars, and weights form a 2D matrix. The transformation is defined as $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

```text title="Traditional Neural Network Shapes"
x: (p)          # p input scalars
W: (q, p)       # mapping p inputs to q outputs
y: (q)          # q output scalars
```

In MATNETS, the fundamental unit of activation is an $n \times n$ matrix. A layer maps a stack of $p$ input matrix-neurons to a stack of $q$ output matrix-neurons.

```text title="MATNETS Shapes"
x: (p, n, n)    # p input matrix-neurons
W: (q, p, n, n) # mapping p inputs to q outputs
B: (q, n, n)    # bias is a full matrix per output neuron
y: (q, n, n)    # q output matrix-neurons
```

Where:

- $p$: The number of input neurons.
- $q$: The number of output neurons.
- $n$: The spatial dimension of the square matrix carried by each neuron.
