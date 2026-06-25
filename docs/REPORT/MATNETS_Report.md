# MATNETS: Matrix-Neuron Neural Networks
### A Research Library for Algebraically-Rich Deep Learning in JAX

> **TL;DR** — MATNETS replaces the scalar neuron with an **n × n matrix neuron**, turning every weight into a matrix-product operator and every activation into a structural transform. Built entirely on JAX primitives; composes with `jit`, `vmap`, and `grad` out of the box. Achieves **~98.5% MNIST accuracy at one-third the parameters** of a standard CNN.

*Repository: [github.com/dsainvg/MATNETS](https://github.com/dsainvg/MATNETS) · Documentation: [dev.dsainvg.me/MATNETS](https://dev.dsainvg.me/MATNETS/) · PyPI: [pypi.org/project/matnets](https://pypi.org/project/matnets/)*

---

## Highlights at a Glance

| | MATNETS |
|---|---|
| **Core idea** | Every neuron is an n × n matrix, not a scalar |
| **Computation** | Full matrix multiplication (not element-wise) |
| **Backend** | Pure JAX — no PyTorch/TF dependency |
| **Transforms** | Fully compatible with `jit`, `vmap`, `grad`, `scan` |
| **Architectures** | Dense · Conv1D/2D · RNN · LSTM · GRU · Attention |
| **MNIST result** | ~98.5% accuracy at **~0.33× the parameters** of StdCNN |
| **Install** | `pip install matnets` |

---

## Installation

```bash
pip install matnets
```

Requires Python ≥ 3.10 and JAX (CPU or GPU). For GPU support, install the appropriate JAX build first:

```bash
# CPU
pip install jax

# CUDA 12 (GPU)
pip install "jax[cuda12]"

# Then install MATNETS
pip install matnets
```

---

## Overview

MATNETS is an experimental JAX library that proposes a fundamental departure from the conventional neural network neuron model. Rather than treating each neuron as a scalar activation, MATNETS endows every neuron with an **n × n matrix**. This seemingly simple shift in representation opens up an entirely new algebraic regime for learning, where the basic unit of computation carries internal structure — orientation, volume, and linear map semantics — that a scalar cannot encode.

---

## The Core Idea: Matrix-Neurons

In a standard dense layer, a neuron holds one number. In MATNETS, a neuron holds an n × n matrix. A layer of `p` input neurons and `q` output neurons therefore works with tensors shaped `(p, n, n)` and `(q, n, n)` respectively, and the weight tensor is `(q, p, n, n)` — a four-dimensional object where every (input neuron, output neuron) pair has an n × n matrix of weights. The bias is similarly matrix-valued, shaped `(q, n, n)`.

The dense primitive computes the following contraction:

> **output = einsum("qpak,pkc → qac", W, x) + B**

This is not Hadamard (element-wise) multiplication — it is full matrix multiplication along the inner indices, meaning each output matrix-neuron is a sum of matrix products of each input neuron's matrix with the corresponding weight matrix, plus a matrix bias. This distinguishes MATNETS from architectures that merely process matrices element-wise; the matrix product semantics are preserved throughout.

---

## Quick Start: Code Examples

### 1 — Dense Layer (drop-in building block)

```python
import jax
import jax.numpy as jnp
import matnets as mtn

# Single matrix-neuron dense forward pass
key = jax.random.key(0)
x = jax.random.normal(key, (4, 2, 2))        # 4 input matrix-neurons, each 2×2

params = mtn.MatrixParams(
    W=jax.random.normal(key, (6, 4, 2, 2)),  # (out_neurons, in_neurons, n, n)
    B=jnp.zeros((6, 2, 2)),                  # (out_neurons, n, n)
)

out = mtn.dense(params, x, activation=jax.nn.relu)
print(out.shape)  # (6, 2, 2) — 6 output matrix-neurons
```

### 2 — Multi-Layer Perceptron (3 hidden layers, Flax + MATNETS)

```python
import flax.linen as nn
import jax, jax.numpy as jnp, optax
from flax.training.train_state import TrainState
import matnets as mtn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, t):
        def layer(name, x, p, q, act=None):
            w = self.param(f"{name}_W", nn.initializers.lecun_normal(), (q, p, 2, 2))
            b = self.param(f"{name}_B", nn.initializers.zeros, (q, 2, 2))
            return mtn.dense(mtn.MatrixParams(W=w, B=b), x, activation=act or (lambda z: z))

        t = layer("l1", t, 2, 6, jax.nn.gelu)
        t = layer("l2", t, 6, 6, jax.nn.gelu)
        t = layer("l3", t, 6, 6, jax.nn.gelu)
        out = layer("out", t, 6, 2)
        return jnp.array([out[0].mean(), out[1].mean()])

model = MLP()
x = jax.random.normal(jax.random.key(0), (64, 2, 2, 2))  # batch of matrix-neuron inputs
params = model.init(jax.random.key(1), x[0])
logits = jax.vmap(lambda t: model.apply(params, t))(x)
print(logits.shape)  # (64, 2)
```

### 3 — 2D Convolutional Network

```python
import flax.linen as nn
import jax, jax.numpy as jnp
import matnets as mtn

class MatCNN2D(nn.Module):
    @nn.compact
    def __call__(self, img):
        # img shape: (H, W, in_channels, n, n)
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, 2, 3, 3, 2, 2)),
            B=self.param("c1_B", nn.initializers.zeros, (4, 2, 2)),
        )
        c2 = mtn.MatrixParams(
            W=self.param("c2_W", nn.initializers.lecun_normal(), (4, 4, 3, 3, 2, 2)),
            B=self.param("c2_B", nn.initializers.zeros, (4, 2, 2)),
        )
        out_p = mtn.MatrixParams(
            W=self.param("out_W", nn.initializers.lecun_normal(), (2, 4, 2, 2)),
            B=self.param("out_B", nn.initializers.zeros, (2, 2, 2)),
        )
        h = jax.nn.relu(mtn.lax.matrix_conv2d(c1, img, padding="SAME"))
        h = jax.nn.relu(mtn.lax.matrix_conv2d(c2, h, padding="SAME"))
        out = mtn.dense(out_p, h.mean(axis=(0, 1)))
        return jnp.array([out[0].mean(), out[1].mean()])
```

### 4 — Matrix-LSTM for Sequence Modelling

```python
import flax.linen as nn
import jax, jax.numpy as jnp
import matnets as mtn
from matnets import nn as mnn

class MatLSTM(nn.Module):
    @nn.compact
    def __call__(self, seq):
        # seq shape: (T, in_neurons, n, n)
        def gate(name):
            return mtn.MatrixParams(
                W=self.param(f"{name}_W", nn.initializers.lecun_normal(), (6, 8, 2, 2)),
                B=self.param(f"{name}_B", nn.initializers.zeros, (6, 2, 2)),
            )

        cell = {"i": gate("i"), "f": gate("f"), "g": gate("g"), "o": gate("o")}
        out_p = mtn.MatrixParams(
            W=self.param("out_W", nn.initializers.lecun_normal(), (2, 6, 2, 2)),
            B=self.param("out_B", nn.initializers.zeros, (2, 2, 2)),
        )
        h0, c0 = jnp.zeros((6, 2, 2)), jnp.zeros((6, 2, 2))
        # jax.lax.scan for JIT-compiled sequence unrolling
        (_, _), hs = jax.lax.scan(lambda carry, x_t: mnn.lstm_step(cell, carry, x_t), (h0, c0), seq)
        out = mtn.dense(out_p, hs[-1], activation=jax.nn.tanh)
        return jnp.array([out[0].mean(), out[1].mean()])
```

---

## Structural Activations and Pooling

MATNETS introduces the idea that activations and pooling operations can either treat the neuron's matrix as a bag of scalars (element-wise, standard) or treat it as a structural unit via its determinant.

**Element-wise activations** (`relu`, `leaky_relu`, `elu`, `sigmoid`, `tanh`, `softplus`) are applied independently to each entry of the n × n matrix, consistent with how standard networks would behave if the matrix were flattened.

**Determinant-gated activations** are novel to this architecture:

- **`relud`** — passes the matrix through only if its determinant is positive; otherwise zeroes it out. This preserves only orientation-preserving linear maps.
- **`leaky_relud`** — like relud, but applies a small scalar (α) to the matrix when det < 0, maintaining gradient flow.
- **`elu_powered`** — for det ≤ 0, applies the matrix exponential branch α(exp(X) − I), which is structurally deep but computationally intensive.

**Determinant-scaled activations** (`elud`, `sigmoidd`, `tanhd`, `softplusd`) instead scale the entire matrix by `fn(det(X)^(1/n)) / det(X)^(1/n)`, using the normalized n-th root of the determinant for dimension-independent stability. This class of activations is smooth and differentiable, treating the matrix's "magnitude" in a geometrically principled way.

**Structural pooling** follows the same philosophy:
- `maxd_pool` selects the matrix with the highest determinant in a window, preserving the winning neuron's full structure.
- `avgd_pool` takes a determinant-weighted average, where each matrix's contribution is modulated by the inverse of its normalized determinant.

Both contrast with standard max/avg pooling, which operate per scalar entry without any awareness of the matrix as a whole.

---

## Full Architecture Suite

Beyond the dense layer, MATNETS has been extended to cover the major architectural families:

**Convolutional layers** (`matrix_conv1d`, `matrix_conv2d`) slide weight tensors of shape `(q, p, kernel_size, n, n)` over matrix-neuron sequences and grids, producing spatially resolved matrix-neuron activations. Standard and determinant-based pooling layers complement these.

**Recurrent layers** (RNN, LSTM, GRU via `matnets.nn`) carry hidden states as stacks of matrices `(hidden_neurons, n, n)`. For LSTM, the gates (input, forget, cell, output) are matrix-valued — meaning the gating mechanism itself operates on n × n objects rather than scalars. Custom gate activations (`sss`, scaled squared sigmoid; `sst`, scaled squared tanh) are also provided, suitable for the matrix regime. All recurrent steps are designed for `jax.lax.scan`, enabling JIT-compiled unrolling over sequences.

**Attention** (`matrix_attention`) generalizes the attention mechanism: queries, keys, and values are token sequences of matrix-neurons, scored using a scaled Frobenius inner product (or a user-supplied score function), and aggregated as matrix-weighted sums. An optional projection through a matrix dense layer can be applied to output tokens.

**Data embedding utilities** (`embed_pixels`, `embed_sequence`) convert conventional scalar inputs into matrix form by extracting local spatial or temporal neighborhoods as n × n windows, providing a natural bridge between standard data pipelines and the matrix-neuron representation.

---

## JAX Integration

MATNETS is designed as a set of pure JAX functions, not as a framework. This means the entire library composes seamlessly with JAX's functional transform ecosystem with no special handling required:

- **`jax.jit`** compiles the einsum-based dense kernel and all other primitives to XLA, producing efficient hardware-specific code.
- **`jax.vmap`** batches any forward function over a batch or token axis without changing the layer definitions — vmap adds the batch dimension transparently around the existing matrix-neuron shapes.
- **`jax.grad`** differentiates through all matrix operations, including the determinant-based activations and pooling, since JAX's automatic differentiation handles these algebraic expressions natively.
- **`jax.lax.scan`** powers the recurrent layers: each step is a closure over matrix-valued carry (H, C for LSTM), and scan unrolls the sequence in a compiled, memory-efficient manner.

Parameters are stored as `MatrixParams`, a named container for W and B that is registered as a JAX pytree. This means parameter trees work directly with `jit`, `vmap`, and `grad` without any custom rules — the standard JAX machinery sees `MatrixParams` just as it would a dict of arrays.

The library deliberately has no dependency on PyTorch or TensorFlow, requiring only JAX and its numpy interface. This keeps the stack minimal and the interop story clean.

---

## Experimental Results: MNIST Benchmark

The attached plot compares four architectures on MNIST — **MatCNN**, **MatCNN0** (a variant without bias initialization warmup), **CapsNet**, and **StdCNN** — across 10 epochs, with the key constraint that the matrix-neuron models use roughly **one-third the parameters** of the standard CNN baseline.

Key observations:

- **MatCNN** reaches ~98% test accuracy by epoch 3 and converges smoothly to ~98.5% by epoch 10, closely tracking the standard CNN despite the parameter reduction.
- **MatCNN0** starts significantly lower (epoch 1 accuracy ~82%, training loss ~1.5), suggesting sensitivity to initialization, but recovers strongly by epoch 3 and converges on par with MatCNN.
- **CapsNet** achieves the best final accuracy (~99%), which is consistent with its routing mechanism's inductive bias for part-whole relationships.
- **StdCNN** performs solidly but does not outperform CapsNet, and closely matches MatCNN.

The central result is that **matrix-neuron CNNs match scalar CNNs at one-third the parameter count on MNIST**, suggesting the richer algebraic structure of matrix neurons packs more information per parameter. The initialization sensitivity of MatCNN0 also points to an interesting open question about how weight initialization should be adapted for the matrix product regime.

---

## Open Research Directions

MATNETS is an early-stage research library, and the results so far open more questions than they close:

- **Determinant-based vs. element-wise activations at scale** — it remains unclear which regime dominates on harder tasks beyond MNIST.
- **Recurrent matrix-neuron dynamics** — matrix-valued LSTM gates carry far more state per neuron; how this interacts with long-range memory is unexplored.
- **Matrix attention at scale** — Frobenius-scored attention over matrix tokens is architecturally novel and has not been benchmarked against transformer baselines.
- **Initialization theory** — Glorot-uniform applied to 4D weight tensors may not be optimal; the n × n matrix product semantics likely demand a rethinking of variance propagation.
- **Computational cost vs. representational gain** — matrix neurons have O(n²) more parameters per neuron; the empirical question is how much more expressive this is relative to simply widening a scalar network.

---

## Summary

MATNETS re-imagines the neuron as a matrix, making matrix multiplication — not scalar weighted addition — the fundamental operation of each layer. The library is built entirely on JAX's functional primitives, composes transparently with all of JAX's transforms, and extends the matrix-neuron idea to dense, convolutional, recurrent, and attention architectures. Initial experiments on MNIST show competitive accuracy with significantly fewer parameters, establishing proof of concept. The theoretical and empirical territory that follows — initialization, scaling, task generalization, the geometric semantics of determinant-based activations — remains wide open.

---

*Repository: [github.com/dsainvg/MATNETS](https://github.com/dsainvg/MATNETS) | Documentation: [dev.dsainvg.me/MATNETS](https://dev.dsainvg.me/MATNETS/) | PyPI: [matnets](https://pypi.org/project/matnets/)*