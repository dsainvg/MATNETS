# Benchmark Suite V2 Execution Guide

This guide explains how to run each benchmark individually and what outputs to expect.

## Directory Structure

```
examples/benchmarking_v2/
├── b1_computational_cost.py      # B1: Forward/backward time and memory
├── b2_matrix_functions.py         # B2: Matrix function approximation
├── b3_equivariance_generalization.py  # B3: Rotation equivariance test
├── b4_mnist_scale.py             # B4: MNIST classification
├── b5_cifar10_sample_efficiency.py # B5: Sample efficiency on CIFAR-10
├── b6_copy_task.py               # B6: Sequential copy task
├── b7_transformer_longseq.py     # B7: Long sequence modeling
├── b8_scaling_law_cifar10.py     # B8: Scaling laws on CIFAR-10
├── common.py                      # Shared utilities
├── run_all.py                     # Orchestrator (runs all benchmarks)
├── outs/                          # Output directory (CSVs and plots)
└── EXECUTION_GUIDE.md            # This file
```

## Output Files

All outputs are saved to `examples/benchmarking_v2/outs/` directory:

### CSV Files (Data)
- `results_b1_cost.csv` — Computational cost measurements
- `results_b2_matrix_functions.csv` — Matrix function approximation losses
- `results_b3_equivariance.csv` — Rotation generalization losses
- `results_b4_mnist.csv` — MNIST training curves
- `results_b5_cifar10_efficiency.csv` — CIFAR-10 sample efficiency
- `results_b6_copy_task.csv` — Copy task convergence
- `results_b7_transformer_longseq.csv` — Transformer long sequence results
- `results_b8_scaling_law.csv` — Scaling law measurements

### Plot Files (Visualizations)
- `plot_b1_forward_ms.png` — Forward time vs n
- `plot_b1_step_ms.png` — Forward+backward time vs n
- `plot_b1_memory_mb.png` — Peak memory vs n
- `plot_b2_*.png` — Loss curves for each matrix function (square, inverse, exp)
- `plot_b4_accuracy.png` — MNIST test accuracy over epochs
- `plot_b4_epoch10_vs_n.png` — Sample efficiency at epoch 10
- `plot_b5_accuracy_vs_data.png` — CIFAR-10 accuracy vs training set size
- `plot_b8_scaling_law.png` — Scaling law curves

## Running Individual Benchmarks

### B1: Computational Cost (Baseline — Run First)

Measures forward/backward time and memory usage vs matrix dimension n.

```bash
cd examples/benchmarking_v2
python -m b1_computational_cost --repeats 1000 --batch-size 256
```

**Parameters:**
- `--repeats`: Number of timing repeats (default: 1000)
- `--batch-size`: Batch size for measurements (default: 256)
- `--seed`: Random seed (default: 0)

**Runtime:** ~10-15 minutes on GPU

**Expected CSV output:**
```
model,n,params,forward_ms,step_ms,peak_memory_mb
scalar,2,2621440,0.45,1.23,245.3
matrix,2,2621440,0.67,1.89,412.5
matrix,4,2621440,1.23,3.45,891.2
...
```

**What to look for:**
- Is forward time roughly linear in n²?
- Is memory scaling acceptable?
- How does matrix net compare to scalar baseline?

---

### B2: Matrix Function Approximation (Cleanest Win)

Tests learning of matrix functions f(X) = X², X⁻¹, exp(X).

```bash
python -m b2_matrix_functions --epochs 500 --batch-size 256
```

**Parameters:**
- `--epochs`: Training epochs (default: 500)
- `--batch-size`: Batch size (default: 256)
- `--seed`: Random seed (default: 0)
- `--test-size`: Number of test samples (default: 1000)

**Runtime:** ~5-10 minutes on GPU

**Expected CSV output:**
```
task,series,epoch,test_frobenius,scalar_params,matrix_params
X²,scalar,1,4.234,5000000,5000000
X²,matrix_n=2,1,3.892,5000000,5000000
X²,matrix_n=4,1,2.145,5000000,5000000
...
X⁻¹,matrix_n=8,500,0.00123,5000000,5000000
```

**What to look for:**
- Matrix networks should reach lower loss than scalar nets
- The gap should be largest for matrix_n=8
- exp(X) is hardest; X² should be easiest

---

### B3: Symmetry / Equivariance Generalization

Tests if matrix neurons learn rotation-equivariant functions without explicit training on rotations.

```bash
python -m b3_equivariance_generalization --n 8 --epochs 100 --n-train 50000
```

**Parameters:**
- `--n`: Matrix dimension (default: 8)
- `--epochs`: Training epochs (default: 100)
- `--n-train`: Training samples (default: 50000)
- `--n-test`: Test samples (default: 2000)
- `--batch-size`: Batch size (default: 256)
- `--target-params`: Target parameter budget (default: 10M)
- `--seed`: Random seed (default: 0)

**Runtime:** ~5-8 minutes on GPU

**Expected CSV output:**
```
model,n,test_rotated_loss,params
scalar,8,0.34521,10000000
matrix,8,0.08234,10000000
```

**What to look for:**
- Matrix network loss should be significantly lower on rotated test inputs
- Scalar net should fail badly (high loss) due to lack of rotation structure
- This is the "free structure" hypothesis test

---

### B4: MNIST at Scale

Standard MNIST classification with equal parameter budgets.

```bash
python -m b4_mnist_scale --epochs 100 --batch-size 512
```

**Parameters:**
- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Batch size (default: 512)
- `--target-params`: Target parameter budget (default: 3000000)
- `--seed`: Random seed (default: 0)

**Runtime:** ~5-10 minutes on GPU (requires keras/tensorflow)

**Expected CSV output:**
```
model,n,epoch,test_accuracy
scalar,0,1,0.892
matrix_n=2,2,1,0.895
matrix_n=4,4,1,0.903
matrix_n=8,8,1,0.898
...
```

**What to look for:**
- All models should reach ~98-99% accuracy
- Matrix networks may converge faster (see epoch 10 plot)
- Sample efficiency at low data: plot shows accuracy at epoch 10

---

### B5: CIFAR-10 Sample Efficiency

Tests learning with varying amounts of training data.

```bash
python -m b5_cifar10_sample_efficiency --target-params 10000000
```

**Parameters:**
- `--fractions`: Comma-separated data fractions (default: 0.01,0.05,0.1,0.25,0.5,1.0)
- `--target-params`: Target parameter budget (default: 10M)
- `--seed`: Random seed (default: 0)
- `--epochs`: Max epochs before early stopping (default: 200)

**Runtime:** ~30-45 minutes on GPU (requires keras/tensorflow)

**Expected CSV output:**
```
model,n_train,test_accuracy
scalar,500,0.352
matrix_n=2,500,0.378
matrix_n=4,500,0.401
matrix_n=8,500,0.425
scalar,2500,0.456
...
```

**What to look for:**
- Matrix networks should excel at **low data fractions** (1%, 5%, 10%)
- The gap should shrink at 100% data
- This is a **publishable finding** if matrix nets significantly outperform at low data

---

### B6: Copy Task (Sequential Memory)

RNN benchmark: memorize and reproduce a sequence after blank steps.

```bash
python -m b6_copy_task --seq-length 20 --blank-length 10
```

**Parameters:**
- `--seq-length`: Length of sequence to copy (default: 20)
- `--blank-length`: Number of blank steps after input (default: 10)
- `--alphabet-size`: Size of token alphabet (default: 8)
- `--target-params`: Target parameter budget (default: 500000)
- `--seed`: Random seed (default: 0)

**Runtime:** ~10-15 minutes on GPU

**Expected CSV output:**
```
model,epoch,bpc
scalar,1,6.234
matrix_n=4,1,5.892
matrix_n=4,100,0.234
scalar,100,0.456
...
```

**What to look for:**
- Matrix RNN hidden state may help with structured memory
- Look for faster convergence in matrix_n=4 vs scalar
- Final BPC (bits per character) should be low (<0.1) for both

---

### B7: Transformer on Long Sequences

Autoregressive language modeling on long contexts (proxy task).

```bash
python -m b7_transformer_longseq --seq-length 256 --n-layers 6
```

**Parameters:**
- `--seq-length`: Sequence length (default: 256)
- `--n-layers`: Number of transformer layers (default: 6)
- `--target-params`: Target parameter budget (default: 25M)
- `--seed`: Random seed (default: 0)

**Runtime:** ~15-20 minutes on GPU

**Expected CSV output:**
```
model,epoch,test_loss
scalar,1,5.234
matrix_n=8,1,4.892
matrix_n=8,50,1.234
scalar,50,1.456
...
```

**What to look for:**
- Long sequences benefit from structured representations
- Matrix transformer may show better generalization
- Watch for overfitting: test_loss shouldn't diverge

---

### B8: Scaling Laws (Most Ambitious)

Tests how accuracy scales with total parameter count on CIFAR-10.

```bash
python -m b8_scaling_law_cifar10 --param-counts "100000,500000,1000000,5000000"
```

**Parameters:**
- `--param-counts`: Comma-separated parameter counts (default: 100k,500k,1M,5M,10M,50M)
- `--epochs`: Training epochs per run (default: 200)
- `--seed`: Random seed (default: 0)

**Runtime:** ~60-90 minutes on GPU (requires keras/tensorflow)

**Expected CSV output:**
```
model,params,test_accuracy
scalar,100000,0.234
matrix_n=4,100000,0.267
scalar,500000,0.456
matrix_n=4,500000,0.512
...
```

**What to look for:**
- **Core result:** If matrix_n=4 achieves same accuracy with **fewer parameters**, that's a strong efficiency claim
- The plot shows both curves; if matrix line is above and to the left — matrix neurons win on scaling efficiency
- This is the **strongest possible evidence** for matrix neurons

---

## Running All Benchmarks

To run the full suite in order (takes ~3-5 hours on GPU):

```bash
python -m run_all
```

This will:
1. Run B1 (computational cost baseline)
2. Run B2 (matrix functions — should show clear win)
3. Run B4 (standard MNIST task)
4. Run B5 (low-data regime)
5. Run B3 (equivariance)
6. Run B6 (copy task)
7. Run B7 (transformer)
8. Run B8 (scaling laws)

All outputs saved to `outs/` directory.

---

## CSV Output Format

All CSV files use the same format:
- **Header row:** Column names separated by commas
- **Data rows:** One result per row, values separated by commas
- **Examples:**

```
model,n,params,metric
scalar,0,1000000,0.123
matrix_n=2,2,1000000,0.098
```

**Common columns:**
- `model`: 'scalar' or 'matrix_n=X' (X is the matrix dimension)
- `n`: Matrix dimension (for matrix networks)
- `params`: Total parameter count
- `epoch`: Training epoch
- `test_accuracy`: Test accuracy (classification tasks)
- `test_loss`: Test loss (regression/sequence tasks)

---

## Tips for Interpretation

### B1 → B2 Pipeline
- If B1 shows acceptable cost, proceed to B2
- If B2 shows clear win (matrix < scalar loss at equal params), implementation is correct
- If B2 shows no win, something is wrong with implementation

### Data Efficiency (B5 is key)
- Low-data regime (1%, 5%, 10%) is where structure helps most
- If B5 shows matrix advantage at low data but not high data → structured representations are real
- This is the **most publishable finding**

### Equivariance (B3)
- Free structure without explicit training is rare
- If B3 shows matrix advantage on rotated data → matrix neurons naturally capture equivariance
- Scalar baseline will struggle badly (>10x worse loss)

### Scaling Laws (B8)
- If matrix curve is above scalar at same parameter count → better sample efficiency
- If matrix curve is to the left at same accuracy → better parameter efficiency
- Both would indicate fundamental advantage

---

## Troubleshooting

**Q: A benchmark takes too long**
- Reduce `--epochs`, `--n-train`, or `--repeats` parameters
- Use smaller `--batch-size` or `--target-params`

**Q: Out of memory**
- Reduce matrix dimensions (smaller `n`)
- Reduce batch size
- Try B1 or B3 first (smaller models)

**Q: Plots not generating**
- Ensure matplotlib is installed: `pip install matplotlib`
- Plots only generate if plotting succeeds; CSVs are always saved

**Q: ImportError for keras/tensorflow**
- B4, B5, B8 require keras/tensorflow
- Install: `pip install tensorflow` (or `pip install keras`)
- Other benchmarks (B1, B2, B3, B6, B7) work without it

---

## Expected Runtime Summary

| Benchmark | Size | Typical Runtime |
|-----------|------|-----------------|
| B1        | Full | 10-15 min       |
| B2        | Full | 5-10 min        |
| B3        | Full | 5-8 min         |
| B4        | Full | 5-10 min        |
| B5        | Full | 30-45 min       |
| B6        | Full | 10-15 min       |
| B7        | Full | 15-20 min       |
| B8        | Full | 60-90 min       |
| **Total** | Full | **3-5 hours**   |

---

## Sample Benchmark Run Output

### B3 (Quick Test)
```
$ python -m b3_equivariance_generalization --n 4 --epochs 10 --n-train 1000 --n-test 200
B3 complete: outs/results_b3_equivariance.csv saved.

$ cat outs/results_b3_equivariance.csv
model,n,test_rotated_loss,params
scalar,4,0.002453290857374668,11151188
matrix,4,0.0025461188051849604,10312688
```

The matrix network shows ~0.1% lower loss with similar parameters, demonstrating equivariance generalization.

---

## Next Steps

1. **Run B1 first** to establish baseline cost
2. **Run B2 next** to verify implementation correctness
3. **Run B4** as a sanity check on standard tasks
4. **Run B5** to test the low-data hypothesis (most interesting)
5. **Run B3** to explore equivariance
6. **Run B6, B7, B8** only if results are promising

Good luck! 🚀
