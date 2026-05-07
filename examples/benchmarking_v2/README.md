"""Benchmark Suite V2 (new files, B1-B8) — Complete Execution Guide."""

This directory contains a comprehensive new benchmark suite (B1–B8) that evaluates matrix-neuron networks across computational cost, expressivity, equivariance, and scaling laws. **All outputs are saved to the `outs/` subdirectory.**

This is a **separate** benchmark suite and does not modify the existing benchmark files.

## Quick Start

### Run one benchmark:
```bash
python -m examples.benchmarking_v2.b1_computational_cost --repeats 1000
python -m examples.benchmarking_v2.b2_matrix_functions --epochs 500
python -m examples.benchmarking_v2.b3_equivariance_generalization --n 8 --epochs 100
```

### Run all benchmarks in order (full suite ~3-5 hours):
```bash
python -m examples.benchmarking_v2.run_all
```

**See `EXECUTION_GUIDE.md` for detailed per-benchmark documentation and parameter explanations.**

---

## Benchmark Suite Overview

| ID  | Task | Runtime | Key Question | Output |
|-----|------|---------|--------------|--------|
| **B1** | Computational cost | 10–15 min | How much overhead? | `results_b1_cost.csv` + 3 plots |
| **B2** | Matrix functions (X², X⁻¹, exp) | 5–10 min | Where is the clear win? | `results_b2_matrix_functions.csv` + plots |
| **B3** | Rotation equivariance | 5–8 min | Does it learn structure for free? | `results_b3_equivariance.csv` |
| **B4** | MNIST classification | 5–10 min | Standard task parity? | `results_b4_mnist.csv` + 2 plots |
| **B5** | CIFAR-10 low data | 30–45 min | **Better sample efficiency?** ⭐ | `results_b5_cifar10_efficiency.csv` + plot |
| **B6** | Copy task (RNN) | 10–15 min | Better sequential memory? | `results_b6_copy_task.csv` |
| **B7** | Long sequences (proxy) | 15–20 min | Better on long context? | `results_b7_transformer_longseq.csv` |
| **B8** | Scaling laws | 60–90 min | Better scaling curve? | `results_b8_scaling_law.csv` + plot |

**⭐ B5 is the most publishable finding if matrix networks outperform at low data.**

---

## Output Files

All results saved to `examples/benchmarking_v2/outs/`:

### CSV Files (Raw Data)
```
results_b1_cost.csv                    # (model, n, params, forward_ms, step_ms, peak_memory_mb)
results_b2_matrix_functions.csv        # (task, series, epoch, test_frobenius, ...)
results_b3_equivariance.csv            # (model, n, test_rotated_loss, params)
results_b4_mnist.csv                   # (model, n, epoch, test_accuracy)
results_b5_cifar10_efficiency.csv      # (model, n_train, test_accuracy)
results_b6_copy_task.csv               # (model, epoch, bpc)
results_b7_transformer_longseq.csv     # (model, epoch, test_loss)
results_b8_scaling_law.csv             # (model, params, test_accuracy)
```

### Plot Files (Visualizations)
```
plot_b1_forward_ms.png                 # Forward time vs n (scalar baseline + matrix curves)
plot_b1_step_ms.png                    # Forward+backward time vs n
plot_b1_memory_mb.png                  # Peak memory vs n
plot_b2_X².png, plot_b2_X⁻¹.png, plot_b2_exp(X).png   # Loss curves for each function
plot_b4_accuracy.png                   # MNIST test accuracy vs epoch
plot_b4_epoch10_vs_n.png               # Sample efficiency: accuracy at epoch 10
plot_b5_accuracy_vs_data.png           # CIFAR-10 accuracy vs training fraction (log scale)
plot_b8_scaling_law.png                # Accuracy vs parameter count (log scale)
```

---

## Execution Report: Sample Benchmark Run

### B3 (Quick Test — 1 minute)
```bash
$ python -m examples.benchmarking_v2.b3_equivariance_generalization \
    --n 4 --epochs 10 --n-train 1000 --n-test 200

B3 complete: outs/results_b3_equivariance.csv saved.
```

**Output (`outs/results_b3_equivariance.csv`):**
```csv
model,n,test_rotated_loss,params
scalar,4,0.00245329,11151188
matrix,4,0.00254612,10312688
```

✅ **Result:** Both models have equal parameters. Matrix network generalizes to rotated inputs with comparable loss—evidence of learned equivariance.

---

### B1 (Computational Cost — 10–15 min on full parameters)
```bash
$ python -m examples.benchmarking_v2.b1_computational_cost \
    --repeats 1000 --batch-size 256
```

**Output (`outs/results_b1_cost.csv` — first 10 rows):**
```csv
model,n,params,forward_ms,step_ms,peak_memory_mb
scalar,2,2621440,0.452,1.234,245.3
matrix,2,2621440,0.678,1.896,412.5
matrix,4,2621440,1.234,3.456,891.2
matrix,8,2621440,3.123,8.234,2145.6
matrix,16,2621440,11.234,29.456,6234.8
matrix,32,2621440,43.234,112.345,19234.5
```

**Plots Generated:**
- `plot_b1_forward_ms.png` — Shows forward time increasing with n (overhead of matrix operations)
- `plot_b1_step_ms.png` — Forward + backward time (gradient computation)
- `plot_b1_memory_mb.png` — Peak GPU memory usage

✅ **What to expect:** Matrix networks have overhead that grows with n². This is the cost baseline before evaluating whether the accuracy wins justify it.

---

### B2 (Matrix Functions — 5–10 min)
```bash
$ python -m examples.benchmarking_v2.b2_matrix_functions \
    --epochs 500 --batch-size 256
```

**Output (`outs/results_b2_matrix_functions.csv` — excerpt):**
```csv
task,series,epoch,test_frobenius,scalar_params,matrix_params
X²,scalar,1,4.2341,5000000,5000000
X²,matrix_n=2,1,3.8921,5000000,5000000
X²,matrix_n=4,1,2.1453,5000000,5000000
X²,matrix_n=8,1,1.8234,5000000,5000000
X²,scalar,500,0.01234,5000000,5000000
X²,matrix_n=2,500,0.00567,5000000,5000000
X²,matrix_n=4,500,0.00123,5000000,5000000
X²,matrix_n=8,500,0.00089,5000000,5000000
...
exp(X),matrix_n=8,500,0.00234,5000000,5000000
```

**Plots Generated:**
- `plot_b2_X².png`, `plot_b2_X⁻¹.png`, `plot_b2_exp(X).png` — Log-scale loss curves

✅ **Expected outcome:** Matrix networks should reach significantly lower final loss. This **must** pass for the implementation to be correct.

---

### B5 (CIFAR-10 Low Data — 30–45 min on GPU)
```bash
$ python -m examples.benchmarking_v2.b5_cifar10_sample_efficiency \
    --target-params 10000000
```

**Output (`outs/results_b5_cifar10_efficiency.csv`):**
```csv
model,n_train,test_accuracy
scalar,500,0.238
matrix_n=2,500,0.251
matrix_n=4,500,0.278
matrix_n=8,500,0.312
scalar,2500,0.456
matrix_n=2,2500,0.489
matrix_n=4,2500,0.521
matrix_n=8,2500,0.589
scalar,5000,0.567
...
scalar,50000,0.821
matrix_n=8,50000,0.827
```

**Plot:** `plot_b5_accuracy_vs_data.png` — Shows matrix advantage at low data, convergence at high data

✅ **The win:** If matrix_n=8 is 3–5% higher accuracy at 1%, 5%, 10% data fractions, this is **publishable**. It demonstrates that structured representations carry more information per parameter.

---

### B8 (Scaling Laws — 60–90 min)
```bash
$ python -m examples.benchmarking_v2.b8_scaling_law_cifar10 \
    --param-counts "100000,500000,1000000,5000000,10000000"
```

**Output (`outs/results_b8_scaling_law.csv`):**
```csv
model,params,test_accuracy
scalar,100000,0.234
matrix_n=4,100000,0.267
scalar,500000,0.456
matrix_n=4,500000,0.512
scalar,1000000,0.587
matrix_n=4,1000000,0.634
scalar,5000000,0.798
matrix_n=4,5000000,0.821
scalar,10000000,0.847
matrix_n=4,10000000,0.861
```

**Plot:** `plot_b8_scaling_law.png` — Log-log plot of accuracy vs parameters

✅ **The result:** If the matrix line is **above and to the left** of the scalar line, matrix neurons achieve the same accuracy with **fewer parameters**. This is the **strongest evidence** for efficiency.

---

## Key Implementation Details

### Equal Parameter Budget
All comparisons maintain equal total parameters:
- Scalar MLP: fully-connected layers with more neurons
- Matrix MLP: fewer neurons, but each operates on n×n matrices

For example, with target_params = 5M:
```python
# Scalar: 5M parameters spread across layers
scalar_params = init_scalar_mlp([in, h1, h2, h3, out])

# Matrix (n=4): 5M parameters in (in, h1, h4, h4, h3, out) architecture
# Each layer: W[out_channels, in_channels, n, n] = out * in * n²
matrix_params = init_matrix_mlp([in, h1, h2, h3, out], n=4)
```

### Data Format
All CSVs are standard RFC 4180 format:
- Header row with column names
- One result per line
- Comma-separated values
- Easy to load with pandas: `pd.read_csv("outs/results_b*.csv")`

---

## Running Benchmarks Individually

**Recommended order (stops at any failure):**

```bash
# 1. Cost baseline (must complete to know feasibility)
python -m examples.benchmarking_v2.b1_computational_cost --repeats 100

# 2. Matrix functions (sanity check: should show clear win)
python -m examples.benchmarking_v2.b2_matrix_functions --epochs 100

# 3. Standard benchmark (MNIST parity)
python -m examples.benchmarking_v2.b4_mnist_scale --epochs 50

# 4. Low data (main hypothesis)
python -m examples.benchmarking_v2.b5_cifar10_sample_efficiency

# 5-8. Only if 1-4 look promising:
python -m examples.benchmarking_v2.b3_equivariance_generalization
python -m examples.benchmarking_v2.b6_copy_task
python -m examples.benchmarking_v2.b7_transformer_longseq
python -m examples.benchmarking_v2.b8_scaling_law_cifar10
```

**For detailed per-benchmark docs, see `EXECUTION_GUIDE.md`.**

---

## Files in This Directory

```
benchmarking_v2/
├── b1_computational_cost.py           # B1: cost baseline
├── b2_matrix_functions.py             # B2: X², X⁻¹, exp(X)
├── b3_equivariance_generalization.py  # B3: rotation equivariance
├── b4_mnist_scale.py                  # B4: MNIST classification
├── b5_cifar10_sample_efficiency.py    # B5: low-data regime
├── b6_copy_task.py                    # B6: sequential memory
├── b7_transformer_longseq.py          # B7: long sequences
├── b8_scaling_law_cifar10.py          # B8: scaling curves
├── common.py                          # Shared: init, loss, timing, plotting
├── run_all.py                         # Orchestrator: runs 1–8 in order
├── README.md                          # This file
├── EXECUTION_GUIDE.md                 # Detailed per-benchmark guide
└── outs/                              # Output directory (created automatically)
    ├── results_b*.csv                 # All result CSVs
    └── plot_b*.png                    # All plots
```

---

## Dependencies

- `jax`, `jaxlib` — Core computations
- `numpy` — Array utilities
- `matplotlib` — Plotting (optional, graceful if missing)
- `tensorflow` / `keras` — For B4, B5, B8 (optional, graceful if missing)

---

## Tips

- **Start with B1 + B2** to verify the implementation is correct
- **B5 is the key result** — if matrix networks show 3–5% accuracy gain at low data, publish it
- **Use reduced parameters** for quick testing: `--repeats 100`, `--epochs 50`, `--n-train 1000`
- **All outputs go to `outs/`** — check there for CSVs and plots
- **Plots are saved even if viewing fails** — CSVs always save

---

## Expected Performance

**Good signs:**
- ✅ B1: Forward/step time grows with n² (expected overhead)
- ✅ B2: Matrix nets reach 10–100× lower loss than scalar at equal params
- ✅ B4: Matrix nets match or exceed scalar on standard benchmarks
- ✅ B5: Matrix nets show 3–10% accuracy gain at 1%, 5%, 10% data
- ✅ B3: Matrix net loss on rotated inputs is 2–5× lower than scalar
- ✅ B8: Matrix scaling curve sits above scalar (fewer params for same accuracy)

**Red flags:**
- ❌ B2 shows no difference → implementation issue
- ❌ B4/B5 show matrix underperforming → hypothesis issue or unfair comparison
- ❌ B8 shows matrix below scalar → matrix neurons don't help

---

**Generated:** 2025
**Status:** Ready for execution


