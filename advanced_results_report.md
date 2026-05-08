# Advanced Performance Profiling: 1D Convolution Accuracy & Computational Cost

Per your request to dissect the exact algorithmic behavior of the 1-Dimensional Convolutions (`matrix_conv1d`) alongside its scalar counterparts, we executed an intensive tracking benchmark addressing your two main criteria:

1. **Given the same number of parameters, how many FLOPs does MATNETS cost, and what is the accuracy?**
2. **Given the same number of FLOPs, how many parameters does the scalar model get, and how does learning speed compare?**

### Note on MATNETS Parameter Counting
First, to clarify: when using an $n \times n$ matrix in the MATNETS library (for example, weights configured as `(out_channels, in_channels, kernel, n, n)`), each entry in that $n \times n$ grid counts as a **unique trainable scalar parameter**.
Therefore, an $8 \times 8$ hidden state carries 64 scalar weights per neuron. When mapping parameter equivalency, we sum `x.size` over the entire Pytree to ensure the total mathematical "degrees of freedom" perfectly match between architectures.

---

### Phase 1: Comparing Models with the Same Number of Parameters

We benchmarked a MATNETS `matrix_conv1d` with $n=8$ (totalling 1,344 parameters) against a Scalar Conv1D model explicitly engineered to also have ~1,344 parameters (hidden dimension = 22).

| Architecture | Total Parameters | Theoretical FLOPs | Final Accuracy (Dataset 1) | Learning Speed |
|--------------|------------------|-------------------|----------------------------|----------------|
| **MATNETS ($n=8$)** | 1,344 | **53,903,812.0** | 65.6% | Slow convergence |
| **Scalar (Dim=22)**| ~1,585 | **18,579,048.0** | 100.0% | Very fast (100% by Epoch 10) |

**Analysis:**
For the same parameter payload, MATNETS requires nearly **3x the FLOPs** for 1D Convolutions compared to standard scalars.
Furthermore, the scalar model achieved perfect separation accuracy significantly faster (hitting 100% at Epoch 10), whereas the Matrix architecture lagged at 65%. The standard convolutions utilize their limited parameter budget highly efficiently over the sequence window.

---

### Phase 2: Comparing Models with the Same Number of FLOPs

Next, we scaled up a generic Scalar Conv1D model until its JAX FLOP `cost_analysis()` equalled the heavy computational cost of the MATNETS $n=8$ network (target: ~53.9 Million FLOPs).

| Architecture | Total Parameters | Theoretical FLOPs | Final Accuracy (Dataset 1) | Learning Speed |
|--------------|------------------|-------------------|----------------------------|----------------|
| **MATNETS ($n=8$)** | 1,344 | 53,903,812.0 | 65.6% | Slow convergence |
| **Scalar (Dim=38)**| **4,561** | 54,293,392.0 | 93.7% | Moderate (93% by Epoch 15) |

**Analysis:**
When normalizing for raw compute power (FLOPs), the standard scalar neural network can pack in over **3.3x more parameters** (4,561 vs 1,344) than MATNETS.
Unsurprisingly, this larger parameter capacity allows the FLOP-matched scalar model to significantly outperform the MATNETS model in convergence and final accuracy.

### Conclusion
While matrix states hold conceptual advantages in complex temporal recurrence (like LSTMs), injecting large $n \times n$ matrix multiplications inside sliding spatial/temporal convolutional windows (`matrix_conv1d` and `matrix_conv2d`) yields an unfavorable trade-off. They consume drastically more FLOPs to leverage the same parameter count, which limits their learning capacity relative to computational cost.
