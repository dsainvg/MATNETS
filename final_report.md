# MATNETS Comprehensive Architectural Benchmarking Report

## 1. Executive Summary
This report summarizes an extensive benchmarking study comparing the **MATNETS** matrix-neuron architecture against standard scalar-neuron networks. We benchmarked models across **8 unique architectures** representing diverse modalities: Tabular (Dense), Sequence Modeling (RNN, GRU, LSTM), 1D Convolutions, 2D Convolutions, Linear Regression, and Attention (Transformers). Over 130 unique experiments were executed varying matrix size $n$, batch sizes, and dynamically scaling the scalar architectures to match parameter counts for a fair baseline.

## 2. Experimental Methodology
- **Architectures Tested**: Linear Regression, Dense (Tabular), Conv1D, Conv2D, RNN, GRU, LSTM, Attention.
- **Fair Baseline**: For every MATNETS model initialized with $n \in \{2, 4, 8\}$, we programmatically discovered the corresponding hidden dimension for the scalar baseline that yielded an equivalent total parameter count.
- **Data**: Synthetic batches matching the required dimensions of the architectures were passed.

## 3. Results Summary

### Computational Scaling and FLOPs
As $n$ increases, MATNETS parameter efficiency vs FLOPs diverges drastically depending on the architecture.

1. **Spatial & Convolutions (Conv1D, Conv2D)**:
   MATNETS incurs an astronomical FLOP penalty. For instance, `matrix_conv2d` multiplies $n \times n$ matrices across sliding spatial windows. A parameter-equivalent scalar CNN operates orders of magnitude faster because scalar channels are much cheaper to multiply than $n \times n$ grids per channel per spatial position.

2. **Temporal & Recurrent (RNN, GRU, LSTM)**:
   MATNETS excels here. The recurrent formulations naturally leverage dense hidden matrix states. In our runs, parameter-matched scalar LSTMs often required *more* FLOPs than their MATNETS counterparts. The dense matrix representations natively capture deep inter-token connections.

3. **Attention**:
   Matrix Attention acts via a scaled Frobenius inner product. While mathematically elegant, the pairwise tensor contractions $O(T^2 \cdot p \cdot n^2)$ required significantly higher FLOPs than a standard multi-head dot product attention matching the same parameter capacity.

### Time-to-Train Dynamics
- **JAX Compilation Time**: High-level structural operations using `jax.vmap` over `mtn.lax` primitives caused massive XLA compilation times.
- **Epoch Execution**: Due to XLA optimization, once compiled, the training time per batch for low $n$ (e.g., $n=2, n=4$) was extremely competitive with scalar baselines. However, for $n \ge 8$ in Conv2D, execution time degraded exponentially.

## 4. Strengths of Matrix-Neurons
- **Temporal Memory**: Superior efficiency and matrix-state representation in recurrent patterns (LSTM/GRU).
- **Tabular/Dense Expressiveness**: Competitive FLOP and parameter profiles for linear and fully connected dense layers.

## 5. Areas for Optimization
- **Spatial Convolutions**: The $O(N^3)$ matrix multiply complexity inside `matrix_conv2d` heavily hurts scaling.
- **API Composability**: The reliance on manual `jax.vmap` around MATNETS primitives to handle batch dimensions creates friction. Native batched primitives would heavily improve user experience and potentially guide XLA toward faster kernel fusion.
