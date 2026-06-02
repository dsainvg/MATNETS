# Changelog

## 3.2.0

- Added three new determinant-scaled activation functions: `sigmoidd`, `tanhd`, and `softplusd`.
- Added `sss` (scaled squared sigmoid) and `sst` (scaled squared tanh) activation functions.
- Renamed the old matrix-exponential determinant-gated ELU to `elu_powered` due to its high computational cost.
- Re-implemented `elud` to use the more efficient and smooth `elu(det(X)^(1/n)) / det(X)^(1/n)` scaling pattern.
- Updated `avgd_pool1d` and `avgd_pool2d` to use `det(M)^(1/n)` scaling instead of raw `det(M)` for dimension-normalized stability.
- Introduced `_safe_det_root` utility with numerical epsilon clamping for robust determinant nth root computation.

## 0.1.0

- Initial project scaffold.
- Added matrix-neuron, layer, and sequential network primitives.
