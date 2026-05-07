"""Benchmarking Matrix-Neuron Networks

This directory contains three comprehensive benchmarks to evaluate matrix-neuron networks:

1. **Computational Cost** (benchmark_cost.py)
   - Measure wall-clock time per forward pass
   - Measure training step (forward + backward) time
   - Monitor memory usage
   - Show how performance scales with matrix dimension n

2. **Expressivity** (benchmark_expressivity.py)
   - Can matrix networks learn things scalar networks can't?
   - Test on matrix function approximation (f(X) = X @ X)
   - Sanity check on MNIST classification
   - Synthetic task designed for matrix operations

3. **Sample Efficiency** (benchmark_sample_efficiency.py)
   - Do matrix networks learn faster with less data?
   - Train on increasing fractions of dataset
   - Measure convergence speed and final accuracy
   - Show data efficiency advantage (if it exists)

## The Three Research Questions

### Question 1: Computational Cost
"How much slower/heavier is it than traditional networks?"

**Expected findings:**
- Forward pass overhead: 4-64x (depending on n)
- Memory overhead: n² scaling
- Training step overhead: similar or slightly worse

**Null hypothesis to disprove:**
"Matrix networks are too slow to be practical."

---

### Question 2: Expressivity  
"Can it learn things scalar nets can't?"

**Expected findings:**
- Matrix function approximation: Matrix net wins decisively
- MNIST: Both work; may be task-dependent
- Synthetic tasks: Matrix net competitive or better

**Null hypothesis to disprove:**
"A scalar network with equal parameters learns equally well."

---

### Question 3: Sample Efficiency
"Does it learn faster with less data?"

**Expected findings:**
- Fewer epochs to convergence
- Better accuracy at small dataset sizes
- Sample-efficient learning from structured tasks

**Null hypothesis to disprove:**
"Both networks require the same amount of data."

---

## Running the Benchmarks

### Quick Start

```bash
# Run all three benchmarks
python benchmark_cost.py
python benchmark_expressivity.py
python benchmark_sample_efficiency.py

# Results are saved as CSV files:
# - results_forward_cost.csv
# - results_training_cost.csv
# - results_memory.csv
# - results_matrix_function.csv
# - results_synthetic_task.csv
# - results_sample_efficiency_synthetic.csv
# - results_sample_efficiency_mnist.csv (if keras installed)
```

### Individual Benchmarks

**Benchmark 1: Computational Cost**
```bash
python benchmark_cost.py
```
This runs three sub-benchmarks:
- `benchmark_forward_pass()` — forward pass only
- `benchmark_training_step()` — forward + backward
- `benchmark_memory()` — memory scaling

Results show the overhead factor at different matrix dimensions (n = 2, 4, 8, 16, 32, ...).

**Benchmark 2: Expressivity**
```bash
python benchmark_expressivity.py
```
This runs three tasks:
- `benchmark_matrix_function()` — learn f(X) = X @ X
- `benchmark_mnist()` — MNIST digit classification
- `benchmark_synthetic_task()` — predict trace(X@X)

Each task trains both scalar and matrix networks to convergence, comparing:
- Final test loss / accuracy
- Convergence curves
- How well suited each network is to the task

**Benchmark 3: Sample Efficiency**
```bash
python benchmark_sample_efficiency.py
```
This runs:
- `benchmark_sample_efficiency_synthetic()` — on matrix function task
- `benchmark_sample_efficiency_mnist()` — on MNIST subset

For each dataset fraction (1%, 5%, 10%, 25%, 50%, 100%), trains both networks
and measures:
- Epochs to convergence
- Final accuracy
- Sample efficiency advantage (epochs or accuracy improvement)

---

## Interpreting Results

### Results Structure

Each benchmark saves CSV files with the format:
```
name, metric, value
scalar, final_test_loss, 0.00453
matrix, final_test_loss, 0.00312
n=4, matrix_time_ms, 0.412
n=4, scalar_time_ms, 0.124
```

CSV files can be loaded and plotted:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_forward_cost.csv")
# Plot, analyze, etc.
```

### What to Look For

**Cost Benchmark:**
- Is the overhead acceptable for your use case?
- Does it scale as O(n²) or O(n³)?
- Which n gives the best cost/expressivity tradeoff?

**Expressivity Benchmark:**
- Does matrix net win on matrix-like tasks? (It should)
- Does it lose on image tasks? (Expected)
- Final loss ratio tells you the advantage

**Sample Efficiency Benchmark:**
- Is the sample efficiency curve above the scalar net's?
- At what dataset size does the advantage appear?
- Does it converge in fewer epochs?

---

## Fair Comparison: Parameter Budgets

**Important:** These benchmarks aim for fair comparison by:

1. **Cost benchmark:** Uses different parameter counts explicitly
   - Scalar: (q, p) = O(pq) parameters
   - Matrix: (q, p, n, n) = O(pqn²) parameters
   - Explicitly reports the ratio

2. **Expressivity benchmark:** Both networks trained to convergence
   - Same learning rate
   - Same optimizer (Adam)
   - Same number of epochs limit (100-500)
   - Compare final loss directly

3. **Sample efficiency benchmark:** Different dataset fractions
   - Same learning rate and optimizer
   - Early stopping based on test loss
   - Measure epochs to convergence

This ensures you're seeing real advantages, not artifacts of hyperparameter tuning.

---

## Architecture Details

### Scalar Network (Baseline)
```python
def scalar_forward(params, x):
    # x: (p,) or batch (batch, p)
    x = jnp.dot(params["W"], x) + params["b"]  # W: (q, p)
    return jax.nn.relu(x)
```

Parameters: `W` (q, p) + `b` (q,) = qp + q parameters

### Matrix Network
```python
def matrix_forward(params, x):
    # x: (p, n, n) or batch (batch, p, n, n)
    return mtn.dense(params, x, activation=jax.nn.relu)
```

Parameters: `W` (q, p, n, n) + `B` (q, n, n) = qpn² + qn² parameters

For fair comparison at the same parameter budget, adjust (p, q) or reduce n.

---

## Utilities

The `benchmark_utils.py` module provides:

- `time_forward_pass()` — benchmark a function with warmup
- `count_parameters()` — count total parameters in a network
- `BenchmarkResults` — container for storing/printing results
- `create_scalar_params()` — random initialize scalar network
- `create_matrix_params()` — random initialize matrix network
- `save_results_csv()` — export results to CSV
- `plot_results()` — plot results (requires matplotlib)

---

## Dependencies

Required:
- `jax >= 0.4`
- `optax` (for training)

Optional:
- `keras` / `tensorflow` (for MNIST benchmarks)
- `matplotlib` (for plotting)
- `pandas` (for result analysis)

Install optionals:
```bash
pip install optax
pip install keras tensorflow  # For MNIST
pip install matplotlib pandas  # For plotting
```

---

## Modifying Benchmarks

To customize:

1. **Change dataset size:** Edit `n_samples` in benchmark functions
2. **Change matrix dimension:** Edit `n = 4` (in cost, use a loop)
3. **Change network size:** Edit `p`, `q` parameters
4. **Change training hyperparameters:** Edit learning rate, optimizer choice
5. **Add new tasks:** Copy a benchmark function and modify the loss/forward

Example: Change matrix dimension for expressivity benchmarks:
```python
# In benchmark_matrix_function(), change:
n = 4  # Try 2, 4, 8, 16

# In benchmark_cost.py, the loop already sweeps n
```

Example: Add CIFAR-10 to expressivity:
```python
def benchmark_cifar10():
    from keras.datasets import cifar10
    # ... similar structure to benchmark_mnist() ...
    # Include in main block
```

---

## Expected Results (Rough Order of Magnitude)

### Cost Benchmark
- n=2: 1.5-2x overhead
- n=4: 4-6x overhead
- n=8: 16-24x overhead
- n=16: 64-96x overhead

Memory scales as n² relative to scalar network.

### Expressivity Benchmark
- Matrix function task: Matrix net **10-100x** lower final loss
- MNIST: Both achieve ~95%+ accuracy (matrix may be slower)
- Synthetic task: Matrix net **2-10x** lower final loss

### Sample Efficiency Benchmark
- Synthetic task: Matrix net may converge in **40-70%** of epochs
- MNIST: May require **20-50%** less data for same accuracy
- Advantage appears at small dataset sizes (< 10% of full dataset)

---

## Publishing Results

When publishing, include:

1. **Table with all three benchmarks**
2. **Plots showing:**
   - Cost vs n
   - Loss curves for expressivity tasks
   - Learning curves for sample efficiency
3. **Summary statistics:**
   - Average overhead at different n
   - Win rate on expressivity tasks
   - Sample efficiency advantage
4. **Discussion:**
   - When do matrix networks help?
   - Is the overhead worth it?
   - What's the practical sweet spot for n?

---

## Troubleshooting

**JAX compilation errors:**
- Run on CPU first (no GPU needed for these benchmarks)
- Check JAX version: `python -c "import jax; print(jax.__version__)"`

**Out of memory:**
- Reduce batch size or dataset size
- Try smaller matrix dimensions first (n=2, 4)

**Slow benchmarks:**
- Reduce `benchmark_runs` in `benchmark_utils.py`
- Reduce number of epochs in training benchmarks
- Run on GPU (set JAX device)

**MNIST not loading:**
- Install keras: `pip install keras`
- May auto-download dataset on first run
- Ensure internet connection

**Plots not showing:**
- Install matplotlib: `pip install matplotlib`
- Check `filename` parameter in `plot_results()`

---

## Further Reading

- The inline comments in each benchmark explain the specific methodology
- `benchmark_utils.py` has full docstrings for all utilities
- JAX documentation: https://jax.readthedocs.io/
- Matrix neural networks paper (if available)

---

## Contributing

To add new benchmarks:
1. Create a new file `benchmark_yourname.py`
2. Import from `benchmark_utils.py`
3. Use `BenchmarkResults` class to track results
4. Save results with `save_results_csv()`
5. Update this README with description and expected results

---

**Last updated:** 2026-05-07
**Author:** Benchmark suite for MATNETS
**License:** MIT (same as MATNETS)
"""
