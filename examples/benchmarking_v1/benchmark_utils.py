"""Utilities for benchmarking matrix-neuron networks."""

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import matnets as mtn
from matnets._params import MatrixParams


def time_forward_pass(
    fn: Callable[[Array], Array],
    x: Array,
    warmup_runs: int = 5,
    benchmark_runs: int = 100,
) -> float:
    """Time a forward pass function.
    
    Args:
        fn: Function to benchmark (should be jitted)
        x: Input array
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        
    Returns:
        Average time per forward pass in milliseconds
    """
    # Warmup
    for _ in range(warmup_runs):
        fn(x).block_until_ready()
    
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(benchmark_runs):
        fn(x).block_until_ready()
    elapsed = time.perf_counter() - t0
    
    return (elapsed / benchmark_runs) * 1000  # Convert to ms


def count_flops_dense_layer(input_size: int, output_size: int, batch_size: int = 1) -> int:
    """Count FLOPs for a dense layer forward pass.
    
    FLOPs = 2 * batch_size * input_size * output_size (multiply-accumulate)
    
    Args:
        input_size: Input dimension (e.g., p)
        output_size: Output dimension (e.g., q)
        batch_size: Batch size
        
    Returns:
        Total FLOPs (multiply-accumulate operations)
    """
    return 2 * batch_size * input_size * output_size


def count_flops_matrix_einsum(p: int, q: int, n: int, batch_size: int = 1) -> int:
    """Count FLOPs for matrix neuron einsum: qpak,pkc->qac
    
    For each of q output neurons:
      For each of p input neurons:
        Matrix multiply (n,n) @ (n,n) -> (n,n): 2*n^3
    Total per forward: q * p * 2 * n^3
    
    Args:
        p: Number of input neurons
        q: Number of output neurons
        n: Matrix dimension
        batch_size: Batch size
        
    Returns:
        Total FLOPs
    """
    # Einsum "qpak,pkc->qac" for single sample:
    # For each q: sum over p of (n,n) * (n,n) multiply -> 2*n^3 ops per (q,p) pair
    flops_per_sample = 2 * q * p * (n ** 3)
    return flops_per_sample * batch_size


def count_flops_backward(flops_forward: int) -> int:
    """Estimate backward pass FLOPs.
    
    Typically backward is 2x forward (gradient w.r.t. inputs and weights).
    
    Args:
        flops_forward: FLOPs for forward pass
        
    Returns:
        Estimated FLOPs for backward pass
    """
    return 2 * flops_forward


def count_parameters(params: dict) -> int:
    """Count total parameters in a parameter dictionary.
    
    Args:
        params: Parameter dict (scalar networks) or MatrixParams (matrix networks)
        
    Returns:
        Total number of parameters
    """
    total = 0
    
    # Handle MatrixParams (frozen dataclass)
    if isinstance(params, MatrixParams):
        total += params.W.size
        total += params.B.size
    # Handle dict (scalar networks)
    elif isinstance(params, dict):
        for v in params.values():
            if isinstance(v, Array):
                total += v.size
            elif isinstance(v, dict):
                total += count_parameters(v)
    
    return total


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        self.results = {}
    
    def add(self, name: str, metric: str, value: float | dict):
        """Add a result.
        
        Args:
            name: Name of the test/configuration
            metric: Metric name (e.g., "time_ms", "accuracy")
            value: Metric value or dict of values
        """
        if name not in self.results:
            self.results[name] = {}
        self.results[name][metric] = value
    
    def get(self, name: str, metric: str | None = None):
        """Get a result.
        
        Args:
            name: Name of the test
            metric: Specific metric, or None to get all metrics
            
        Returns:
            Result value or dict of results
        """
        if metric is None:
            return self.results.get(name, {})
        return self.results.get(name, {}).get(metric)
    
    def summary(self) -> str:
        """Print a summary of all results."""
        lines = ["Benchmark Results:"]
        for name, metrics in self.results.items():
            lines.append(f"\n{name}:")
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        lines.append(f"  {metric}[{k}]: {v:.4f}")
                else:
                    lines.append(f"  {metric}: {value:.4f}")
        return "\n".join(lines)


def create_scalar_params(
    key: jax.random.PRNGKey,
    p: int,
    q: int,
    scale: float = 0.01,
) -> dict:
    """Create parameters for a scalar network with equivalent structure.
    
    Args:
        key: JAX random key
        p: Input dimension (in terms of neurons)
        q: Output dimension (in terms of neurons)
        scale: Weight initialization scale
        
    Returns:
        Parameter dictionary with 'W' and 'b'
    """
    key_w, key_b = jax.random.split(key)
    W = jax.random.normal(key_w, (q, p)) * scale
    b = jax.random.normal(key_b, (q,)) * scale
    return {"W": W, "b": b}


def create_matrix_params(
    key: jax.random.PRNGKey,
    p: int,
    q: int,
    n: int,
    scale: float = 0.01,
) -> MatrixParams:
    """Create parameters for a matrix-neuron network.
    
    Args:
        key: JAX random key
        p: Input dimension (number of input neurons)
        q: Output dimension (number of output neurons)
        n: Matrix dimension
        scale: Weight initialization scale
        
    Returns:
        MatrixParams object with W and B
    """
    key_w, key_b = jax.random.split(key)
    W = jax.random.normal(key_w, (q, p, n, n)) * scale
    B = jax.random.normal(key_b, (q, n, n)) * scale
    return MatrixParams(W=W, B=B)


def save_results_csv(results: BenchmarkResults, filename: str):
    """Save results to CSV format.
    
    Args:
        results: BenchmarkResults object
        filename: Output filename
    """
    import csv
    import os
    
    # Ensure outs directory exists
    os.makedirs("outs", exist_ok=True)
    
    # Prepend outs/ to filename
    filepath = os.path.join("outs", filename)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "metric", "value"])
        
        for name, metrics in results.results.items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        writer.writerow([name, f"{metric}_{k}", v])
                else:
                    writer.writerow([name, metric, value])


def plot_results(
    results: BenchmarkResults,
    x_key: str,
    y_keys: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str | None = None,
):
    """Plot benchmark results.
    
    Requires matplotlib.
    
    Args:
        results: BenchmarkResults object
        x_key: Key for x-axis metric
        y_keys: List of keys for y-axis metrics
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Optional filename to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for y_key in y_keys:
        x_vals = []
        y_vals = []
        
        for name, metrics in results.results.items():
            if x_key in metrics and y_key in metrics:
                x_vals.append(metrics[x_key])
                y_vals.append(metrics[y_key])
        
        if x_vals and y_vals:
            # Sort by x values
            sorted_pairs = sorted(zip(x_vals, y_vals))
            x_vals = [p[0] for p in sorted_pairs]
            y_vals = [p[1] for p in sorted_pairs]
            ax.plot(x_vals, y_vals, marker='o', label=y_key)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()


def create_matched_scalar_params(
    key: jax.random.PRNGKey,
    target_params: int,
    q_out: int,
    scale: float = 0.01,
) -> tuple[dict, int, int]:
    """Create scalar network parameters matched to target parameter budget.
    
    Given a target parameter count and output dimension, find input dimension p
    such that q*p + q ≈ target_params.
    
    Args:
        key: JAX random key
        target_params: Target number of parameters
        q_out: Output dimension
        scale: Weight initialization scale
        
    Returns:
        Tuple of (params_dict, p, q) where params_dict has parameters matched to budget
    """
    # q*p + q ≈ target_params
    # p ≈ (target_params - q) / q
    p = max(1, (target_params - q_out) // q_out)
    
    key_w, key_b = jax.random.split(key)
    W = jax.random.normal(key_w, (q_out, p)) * scale
    b = jax.random.normal(key_b, (q_out,)) * scale
    
    actual_params = q_out * p + q_out
    return {"W": W, "b": b}, p, q_out


def format_number(num: int | float | str) -> str:
    """Format a number with commas and appropriate precision.
    
    Args:
        num: Number to format or string to pass through
        
    Returns:
        Formatted string
    """
    if isinstance(num, str):
        return num
    if isinstance(num, float):
        if num < 1.0:
            return f"{num:.6f}"
        elif num < 1000:
            return f"{num:.2f}"
        else:
            return f"{num:,.0f}"
    return f"{num:,}"


def print_network_comparison(
    network_type: str,
    architecture: dict,
    params: int,
    flops_forward: int,
    flops_backward: int,
    memory_bytes: int,
    time_ms: float = None,
    accuracy: float = None,
    loss: float = None,
):
    """Pretty-print network comparison details.
    
    Args:
        network_type: "Scalar" or "Matrix"
        architecture: Dict with architecture details
        params: Total parameters
        flops_forward: FLOPs for forward pass
        flops_backward: FLOPs for backward pass
        memory_bytes: Memory usage in bytes
        time_ms: Optional wall-clock time in ms
        accuracy: Optional test accuracy
        loss: Optional test loss
    """
    print(f"\n{network_type} Network:")
    print(f"  Architecture: {architecture}")
    print(f"  Parameters:    {format_number(params):>15}")
    print(f"  FLOPs/forward: {format_number(flops_forward):>15}")
    print(f"  FLOPs/backward:{format_number(flops_backward):>15}")
    print(f"  FLOPs/step:    {format_number(flops_forward + flops_backward):>15}")
    print(f"  Memory (MB):   {memory_bytes / (1024**2):>15.3f}")
    
    if time_ms is not None:
        print(f"  Time (ms):     {time_ms:>15.4f}")
    if accuracy is not None:
        print(f"  Accuracy:      {accuracy:>15.4f}")
    if loss is not None:
        print(f"  Loss:          {loss:>15.6f}")


def print_comparison_table(
    comparison_list: list[dict],
    title: str = "Network Comparison",
):
    """Print a formatted table comparing multiple networks.
    
    Args:
        comparison_list: List of dicts with keys:
            - name: Network name
            - params: Parameter count
            - flops: FLOPs per forward
            - time_ms: Wall-clock time (optional)
            - accuracy: Test accuracy (optional)
            - loss: Test loss (optional)
        title: Table title
    """
    print("\n" + "="*120)
    print(f" {title}".ljust(120))
    print("="*120)
    
    # Header
    header = ["Network", "Params", "FLOPs (fwd)", "Time (ms)", "Accuracy", "Loss"]
    widths = [25, 15, 20, 15, 15, 15]
    header_line = ""
    for h, w in zip(header, widths):
        header_line += f"{h:^{w}}"
    print(header_line)
    print("-"*120)
    
    # Rows
    for item in comparison_list:
        row = ""
        row += f"{item.get('name', ''):^25}"
        row += f"{format_number(item.get('params', 0)):>15}"
        row += f"{format_number(item.get('flops', 0)):>20}"
        
        time_str = f"{item.get('time_ms', 0):.4f}" if item.get('time_ms') else "N/A"
        row += f"{time_str:>15}"
        
        acc_str = f"{item.get('accuracy', 0):.4f}" if item.get('accuracy') else "N/A"
        row += f"{acc_str:>15}"
        
        loss_str = f"{item.get('loss', 0):.6f}" if item.get('loss') else "N/A"
        row += f"{loss_str:>15}"
        
        print(row)
    
    print("="*120)
