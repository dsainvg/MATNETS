"""Benchmark 1: Computational Cost

Measure:
- Wall clock time per forward pass
- Wall clock time per training step (forward + backward)
- Memory usage
- How costs scale with n (matrix dimension)

Fair comparison: Equal parameter budget between scalar and matrix networks.
This benchmark now shows:
- Exact parameter counts
- Exact FLOP counts (multiply-accumulate operations)
- Wall-clock time comparisons
- Memory usage
- Detailed pretty-printed tables
"""

import jax
import jax.numpy as jnp
import time
from jax import Array
from jax.experimental import io_callback

import matnets as mtn
from matnets._params import MatrixParams
from examples.benchmarking_v1.benchmark_utils import (
    BenchmarkResults,
    count_parameters,
    count_flops_dense_layer,
    count_flops_matrix_einsum,
    count_flops_backward,
    create_scalar_params,
    create_matrix_params,
    create_matched_scalar_params,
    time_forward_pass,
    save_results_csv,
    plot_results,
    print_network_comparison,
    print_comparison_table,
)


def scalar_forward(params: dict, x: Array) -> Array:
    """Scalar network forward pass.
    
    params: {'W': (q, p), 'b': (q,)}
    x: (p,) scalar vector
    """
    x = jnp.dot(params["W"], x) + params["b"]
    return jax.nn.relu(x)


def scalar_forward_batch(params: dict, x: Array) -> Array:
    """Scalar network forward pass on batch.
    
    params: {'W': (q, p), 'b': (q,)}
    x: (batch, p) scalar vectors
    """
    x = jnp.dot(x, params["W"].T) + params["b"]
    return jax.nn.relu(x)


def matrix_forward(params: MatrixParams, x: Array) -> Array:
    """Matrix-neuron network forward pass.
    
    params: MatrixParams with W: (q, p, n, n), B: (q, n, n)
    x: (p, n, n) matrix input
    """
    return mtn.dense(params, x, activation=jax.nn.relu)


def matrix_forward_batch(params: MatrixParams, x: Array) -> Array:
    """Matrix-neuron network forward pass on batch.
    
    params: MatrixParams with W: (q, p, n, n), B: (q, n, n)
    x: (batch, p, n, n) matrix inputs
    
    Use vmap to vectorize over batch dimension.
    """
    def single_forward(xi):
        return mtn.dense(params, xi, activation=jax.nn.relu)
    
    return jax.vmap(single_forward)(x)


def benchmark_forward_pass():
    """Benchmark forward pass computational cost.
    
    Sweeps over matrix dimension n while keeping parameter budget fixed.
    Matches scalar network parameters to matrix network for fair comparison.
    """
    print("\n" + "="*100)
    print("BENCHMARK 1A: Forward Pass Computational Cost")
    print("="*100)
    
    results = BenchmarkResults()
    
    # Control parameters
    q = 4  # output neurons
    batch_size = 32
    
    # For each matrix dimension n
    for n in [2, 4, 8, 16]:
        print(f"\n{'='*100}")
        print(f"Matrix dimension n = {n}")
        print(f"{'='*100}")
        
        # Create matrix network with some p
        p = 4  # input neurons
        matrix_params = create_matrix_params(jax.random.key(0), p, q, n, scale=0.01)
        matrix_params_count = count_parameters(matrix_params)
        
        print(f"\nMatrix Network: p={p}, q={q}, n={n}")
        print(f"  W shape: ({q}, {p}, {n}, {n})")
        print(f"  B shape: ({q}, {n}, {n})")
        print(f"  Total parameters: {matrix_params_count:,}")
        
        # FLOPs for matrix network
        flops_matrix_fwd = count_flops_matrix_einsum(p, q, n, batch_size=1)
        flops_matrix_bwd = count_flops_backward(flops_matrix_fwd)
        memory_matrix = matrix_params_count * 4  # float32
        
        print(f"  FLOPs/forward:  {flops_matrix_fwd:,}")
        print(f"  FLOPs/backward: {flops_matrix_bwd:,}")
        print(f"  Memory (bytes): {memory_matrix:,}")
        
        # Create matched scalar network
        scalar_params, p_scalar, q_scalar = create_matched_scalar_params(
            jax.random.key(0),
            target_params=matrix_params_count,
            q_out=q,
        )
        scalar_params_count = count_parameters(scalar_params)
        
        print(f"\nScalar Network (matched to {matrix_params_count:,} parameters)")
        print(f"  p={p_scalar}, q={q_scalar}")
        print(f"  W shape: ({q_scalar}, {p_scalar})")
        print(f"  b shape: ({q_scalar},)")
        print(f"  Total parameters: {scalar_params_count:,}")
        
        # FLOPs for scalar network
        flops_scalar_fwd = count_flops_dense_layer(p_scalar, q_scalar, batch_size=1)
        flops_scalar_bwd = count_flops_backward(flops_scalar_fwd)
        memory_scalar = scalar_params_count * 4
        
        print(f"  FLOPs/forward:  {flops_scalar_fwd:,}")
        print(f"  FLOPs/backward: {flops_scalar_bwd:,}")
        print(f"  Memory (bytes): {memory_scalar:,}")
        
        # Input shapes
        x_scalar = jnp.zeros((batch_size, p_scalar))
        x_matrix = jnp.zeros((batch_size, p, n, n))
        
        # Benchmark scalar forward
        def scalar_forward_batch(params, x):
            x = jnp.dot(x, params["W"].T) + params["b"]
            return jax.nn.relu(x)
        
        scalar_jit = jax.jit(scalar_forward_batch)
        scalar_time = time_forward_pass(
            lambda x: scalar_jit(scalar_params, x),
            x_scalar,
            warmup_runs=5,
            benchmark_runs=100,
        )
        
        # Benchmark matrix forward
        def matrix_forward_batch(params, x):
            def single_forward(xi):
                return mtn.dense(params, xi, activation=jax.nn.relu)
            return jax.vmap(single_forward)(x)
        
        matrix_jit = jax.jit(matrix_forward_batch)
        matrix_time = time_forward_pass(
            lambda x: matrix_jit(matrix_params, x),
            x_matrix,
            warmup_runs=5,
            benchmark_runs=100,
        )
        
        overhead = matrix_time / scalar_time
        flops_overhead = flops_matrix_fwd / flops_scalar_fwd
        
        print(f"\n--- Performance Comparison ---")
        print(f"Scalar forward time: {scalar_time:.4f} ms")
        print(f"Matrix forward time: {matrix_time:.4f} ms")
        print(f"Time overhead: {overhead:.2f}x")
        print(f"FLOP overhead (theoretical): {flops_overhead:.2f}x")
        
        results.add(f"n={n}", "scalar_time_ms", scalar_time)
        results.add(f"n={n}", "matrix_time_ms", matrix_time)
        results.add(f"n={n}", "time_overhead", overhead)
        results.add(f"n={n}", "flops_overhead", flops_overhead)
        results.add(f"n={n}", "scalar_flops_fwd", flops_scalar_fwd)
        results.add(f"n={n}", "matrix_flops_fwd", flops_matrix_fwd)
        results.add(f"n={n}", "scalar_params", scalar_params_count)
        results.add(f"n={n}", "matrix_params", matrix_params_count)
        results.add(f"n={n}", "n", n)
        
        # Pretty print comparison
        print_comparison_table([
            {
                "name": f"Scalar (p={p_scalar})",
                "params": scalar_params_count,
                "flops": flops_scalar_fwd,
                "time_ms": scalar_time,
            },
            {
                "name": f"Matrix (n={n})",
                "params": matrix_params_count,
                "flops": flops_matrix_fwd,
                "time_ms": matrix_time,
            },
        ], title=f"Forward Pass Comparison (n={n})")
    
    print("\n" + results.summary())
    save_results_csv(results, "results_forward_cost.csv")
    return results


def benchmark_training_step():
    """Benchmark training step (forward + backward) cost."""
    print("\n" + "="*100)
    print("BENCHMARK 1B: Training Step (Forward + Backward) Computational Cost")
    print("="*100)
    
    results = BenchmarkResults()
    
    q = 4
    batch_size = 32
    
    # Create loss functions
    def scalar_loss(params, x, y):
        def scalar_forward(params, x):
            x = jnp.dot(params["W"], x) + params["b"]
            return jax.nn.relu(x)
        pred = jax.vmap(lambda xi: scalar_forward(params, xi))(x)
        return jnp.mean((pred - y) ** 2)
    
    def matrix_loss(params, x, y):
        def matrix_forward(params, x):
            return mtn.dense(params, x, activation=jax.nn.relu)
        pred = jax.vmap(matrix_forward, in_axes=(None, 0))(params, x)
        return jnp.mean((pred - y) ** 2)
    
    # Scalar network
    p_scalar = 4
    scalar_params = create_scalar_params(jax.random.key(0), p_scalar, q, scale=0.01)
    scalar_loss_jit = jax.jit(scalar_loss)
    scalar_grad_jit = jax.jit(jax.grad(scalar_loss))
    
    scalar_params_count = count_parameters(scalar_params)
    flops_scalar_fwd = count_flops_dense_layer(p_scalar, q, batch_size=batch_size)
    flops_scalar_total = flops_scalar_fwd + count_flops_backward(flops_scalar_fwd)
    
    x_scalar = jax.random.normal(jax.random.key(0), (batch_size, p_scalar))
    y_scalar = jax.random.normal(jax.random.key(1), (batch_size, q))
    
    # Warm up and time scalar training step
    def scalar_step(params, x, y):
        loss = scalar_loss_jit(params, x, y)
        grads = scalar_grad_jit(params, x, y)
        return loss, grads
    
    t0 = time.perf_counter()
    for _ in range(50):
        scalar_step(scalar_params, x_scalar, y_scalar)
        jax.effects_barrier()
    scalar_train_time = (time.perf_counter() - t0) / 50 * 1000
    
    print(f"\n{'='*100}")
    print("Scalar Network Baseline")
    print(f"{'='*100}")
    print(f"Parameters:     {scalar_params_count:,}")
    print(f"FLOPs/forward:  {flops_scalar_fwd:,}")
    print(f"FLOPs/backward: {count_flops_backward(flops_scalar_fwd):,}")
    print(f"FLOPs/step:     {flops_scalar_total:,}")
    print(f"Training step time: {scalar_train_time:.4f} ms")
    
    results.add("scalar", "train_step_ms", scalar_train_time)
    results.add("scalar", "params", scalar_params_count)
    results.add("scalar", "flops_total", flops_scalar_total)
    
    # Matrix network
    for n in [2, 4, 8, 16]:
        print(f"\n{'='*100}")
        print(f"Matrix Network (n={n})")
        print(f"{'='*100}")
        
        p = 4
        matrix_params = create_matrix_params(jax.random.key(0), p, q, n, scale=0.01)
        matrix_params_count = count_parameters(matrix_params)
        
        matrix_loss_jit = jax.jit(matrix_loss)
        matrix_grad_jit = jax.jit(jax.grad(matrix_loss))
        
        x_matrix = jax.random.normal(jax.random.key(0), (batch_size, p, n, n))
        y_matrix = jax.random.normal(jax.random.key(1), (batch_size, q, n, n))
        
        # FLOPs
        flops_matrix_fwd = count_flops_matrix_einsum(p, q, n, batch_size=batch_size)
        flops_matrix_total = flops_matrix_fwd + count_flops_backward(flops_matrix_fwd)
        
        print(f"Parameters:     {matrix_params_count:,}")
        print(f"FLOPs/forward:  {flops_matrix_fwd:,}")
        print(f"FLOPs/backward: {count_flops_backward(flops_matrix_fwd):,}")
        print(f"FLOPs/step:     {flops_matrix_total:,}")
        
        def matrix_step(params, x, y):
            loss = matrix_loss_jit(params, x, y)
            grads = matrix_grad_jit(params, x, y)
            return loss, grads
        
        t0 = time.perf_counter()
        for _ in range(50):
            matrix_step(matrix_params, x_matrix, y_matrix)
            jax.effects_barrier()
        matrix_train_time = (time.perf_counter() - t0) / 50 * 1000
        
        overhead = matrix_train_time / scalar_train_time
        flops_overhead = flops_matrix_total / flops_scalar_total
        
        print(f"Training step time: {matrix_train_time:.4f} ms")
        print(f"Time overhead: {overhead:.2f}x")
        print(f"FLOP overhead (theoretical): {flops_overhead:.2f}x")
        
        results.add(f"n={n}", "train_step_ms", matrix_train_time)
        results.add(f"n={n}", "overhead", overhead)
        results.add(f"n={n}", "flops_overhead", flops_overhead)
        results.add(f"n={n}", "params", matrix_params_count)
        results.add(f"n={n}", "flops_total", flops_matrix_total)
        results.add(f"n={n}", "n", n)
        
        # Pretty print comparison
        print_comparison_table([
            {
                "name": f"Scalar",
                "params": scalar_params_count,
                "flops": flops_scalar_total,
                "time_ms": scalar_train_time,
            },
            {
                "name": f"Matrix (n={n})",
                "params": matrix_params_count,
                "flops": flops_matrix_total,
                "time_ms": matrix_train_time,
            },
        ], title=f"Training Step Comparison (n={n})")
    
    print("\n" + results.summary())
    save_results_csv(results, "results_training_cost.csv")
    return results


def benchmark_memory():
    """Benchmark memory usage with increasing matrix dimensions.
    
    Shows exact parameter counts and FLOPs for each configuration.
    """
    print("\n" + "="*100)
    print("BENCHMARK 1C: Memory Usage and FLOP Scaling with Increasing n")
    print("="*100)
    
    results = BenchmarkResults()
    
    p = 4
    q = 4
    batch_size = 32
    
    # Scalar baseline
    scalar_params = create_scalar_params(jax.random.key(0), p, q)
    scalar_params_count = count_parameters(scalar_params)
    scalar_memory_bytes = scalar_params_count * 4  # float32
    flops_scalar = count_flops_dense_layer(p, q, batch_size=batch_size)
    
    print(f"\n{'='*100}")
    print("Scalar Network Baseline")
    print(f"{'='*100}")
    print(f"Architecture: p={p}, q={q}")
    print(f"Parameters:   {scalar_params_count:,}")
    print(f"Memory (MB):  {scalar_memory_bytes / (1024**2):.3f}")
    print(f"FLOPs/step:   {flops_scalar:,}")
    
    results.add("scalar", "memory_bytes", scalar_memory_bytes)
    results.add("scalar", "params", scalar_params_count)
    results.add("scalar", "flops_per_step", flops_scalar)
    
    comparison_rows = [
        {
            "name": "Scalar",
            "params": scalar_params_count,
            "flops": flops_scalar,
        }
    ]
    
    # Matrix networks with increasing n
    for n in [2, 4, 8, 16, 32, 64]:
        matrix_params = create_matrix_params(jax.random.key(0), p, q, n)
        matrix_params_count = count_parameters(matrix_params)
        matrix_memory_bytes = matrix_params_count * 4
        flops_matrix = count_flops_matrix_einsum(p, q, n, batch_size=batch_size)
        
        ratio_params = matrix_params_count / scalar_params_count
        ratio_memory = matrix_memory_bytes / scalar_memory_bytes
        ratio_flops = flops_matrix / flops_scalar
        
        print(f"\n{'='*100}")
        print(f"Matrix Network (n={n})")
        print(f"{'='*100}")
        print(f"Architecture: p={p}, q={q}, n={n}")
        print(f"W shape: ({q}, {p}, {n}, {n})")
        print(f"B shape: ({q}, {n}, {n})")
        print(f"Parameters:      {matrix_params_count:,}  ({ratio_params:.1f}x scalar)")
        print(f"Memory (MB):     {matrix_memory_bytes / (1024**2):.3f}  ({ratio_memory:.1f}x scalar)")
        print(f"FLOPs/step:      {flops_matrix:,}  ({ratio_flops:.1f}x scalar)")
        
        results.add(f"n={n}", "memory_bytes", matrix_memory_bytes)
        results.add(f"n={n}", "memory_ratio_vs_scalar", ratio_memory)
        results.add(f"n={n}", "params", matrix_params_count)
        results.add(f"n={n}", "params_ratio_vs_scalar", ratio_params)
        results.add(f"n={n}", "flops_per_step", flops_matrix)
        results.add(f"n={n}", "flops_ratio_vs_scalar", ratio_flops)
        results.add(f"n={n}", "n", n)
        
        comparison_rows.append({
            "name": f"Matrix (n={n})",
            "params": matrix_params_count,
            "flops": flops_matrix,
        })
    
    # Print summary table
    print_comparison_table(comparison_rows, title="Memory and FLOP Summary (all n)")
    
    print("\n" + results.summary())
    save_results_csv(results, "results_memory.csv")
    return results


if __name__ == "__main__":
    print("COMPUTATIONAL COST BENCHMARKS")
    print("="*70)
    print("\nThese benchmarks measure the overhead of matrix-neuron networks")
    print("compared to scalar networks at various matrix dimensions.")
    
    results_forward = benchmark_forward_pass()
    results_training = benchmark_training_step()
    results_memory = benchmark_memory()
    
    print("\n" + "="*70)
    print("Cost benchmark complete. Results saved to:")
    print("  - outs/results_forward_cost.csv")
    print("  - outs/results_training_cost.csv")
    print("  - outs/results_memory.csv")
    print("="*70)
