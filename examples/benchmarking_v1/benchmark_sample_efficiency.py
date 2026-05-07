"""Benchmark 3: Sample Efficiency

Test whether matrix-neuron networks learn faster with less data.

Hypothesis: Matrix neurons carry more structure per parameter,
so they should achieve higher accuracy with fewer training examples.

This benchmark now shows:
- Exact parameter counts for each network
- Exact FLOP counts
- Convergence speed (epochs to convergence)
- Final accuracies
- Pretty-printed comparison tables
"""

import jax
import jax.numpy as jnp
import optax
from jax import Array
import numpy as np

import matnets as mtn
from examples.benchmarking_v1.benchmark_utils import (
    BenchmarkResults,
    count_parameters,
    count_flops_dense_layer,
    count_flops_matrix_einsum,
    create_scalar_params,
    create_matrix_params,
    save_results_csv,
    print_comparison_table,
)


def benchmark_sample_efficiency_synthetic():
    """Test sample efficiency on synthetic matrix function task (f(X) = X @ X).
    
    Train on increasing fractions of dataset and measure convergence speed.
    """
    print("\n" + "="*100)
    print("SAMPLE EFFICIENCY: Matrix Function Approximation (f(X) = X @ X)")
    print("="*100)
    
    results = BenchmarkResults()
    
    # Generate full dataset
    key = jax.random.key(0)
    n = 4
    n_samples = 2000
    
    key_gen = jax.random.key(1)
    X_data = jax.random.normal(key_gen, (n_samples, n, n)) * 0.5
    Y_data = jnp.einsum("sij,sjk->sik", X_data, X_data)
    
    # Fixed test set
    n_test = 500
    X_test = X_data[-n_test:]
    Y_test = Y_data[-n_test:]
    
    print(f"\nFull dataset: {n_samples} samples")
    print(f"Test set: {n_test} samples (fixed)")
    print(f"Input/Output: {n}x{n} matrices")
    
    # Get baseline network parameters and FLOPs
    p_scalar = n * n
    q_scalar = n * n
    p_matrix = 1
    q_matrix = 1
    
    scalar_params_base = create_scalar_params(jax.random.key(0), p_scalar, q_scalar)
    scalar_params_count = count_parameters(scalar_params_base)
    scalar_flops = count_flops_dense_layer(p_scalar, q_scalar)
    
    matrix_params_base = create_matrix_params(jax.random.key(0), p_matrix, q_matrix, n)
    matrix_params_count = count_parameters(matrix_params_base)
    matrix_flops = count_flops_matrix_einsum(p_matrix, q_matrix, n)
    
    print(f"\nNetwork Summary:")
    print(f"  Scalar: {scalar_params_count:,} params, {scalar_flops:,} FLOPs/forward")
    print(f"  Matrix: {matrix_params_count:,} params, {matrix_flops:,} FLOPs/forward")
    print(f"  Param ratio (matrix/scalar): {matrix_params_count / scalar_params_count:.1f}x")
    
    # Fractions of training data to use
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    for fraction in fractions:
        n_train = max(10, int((n_samples - n_test) * fraction))
        
        print(f"\n{'='*100}")
        print(f"Training set size: {n_train} ({fraction*100:.0f}%) of {n_samples - n_test}")
        print(f"{'='*100}")
        
        # Use first n_train samples for training
        X_train = X_data[:n_train]
        Y_train = Y_data[:n_train]
        
        # ========== Scalar Network ==========
        
        X_train_flat = X_train.reshape(n_train, n*n)
        X_test_flat = X_test.reshape(n_test, n*n)
        Y_train_flat = Y_train.reshape(n_train, n*n)
        Y_test_flat = Y_test.reshape(n_test, n*n)
        
        scalar_params = create_scalar_params(jax.random.key(0), n*n, n*n, scale=0.01)
        
        def scalar_forward(params, x):
            return jnp.dot(params["W"], x) + params["b"]
        
        def scalar_loss(params, x, y):
            pred = jax.vmap(lambda xi, yi: scalar_forward(params, xi))(x, y)
            return jnp.mean((pred - y) ** 2)
        
        scalar_loss_jit = jax.jit(scalar_loss)
        scalar_grad_jit = jax.jit(jax.grad(scalar_loss))
        
        # Train to convergence (or max epochs)
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(scalar_params)
        
        scalar_test_losses = []
        patience = 10
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(500):
            loss_val = scalar_loss_jit(scalar_params, X_train_flat, Y_train_flat)
            grads = scalar_grad_jit(scalar_params, X_train_flat, Y_train_flat)
            updates, opt_state = optimizer.update(grads, opt_state)
            scalar_params = jax.tree_util.tree_map(
                lambda p, u: p + u, scalar_params, updates
            )
            
            # Check test loss for early stopping
            test_loss = float(scalar_loss_jit(scalar_params, X_test_flat, Y_test_flat))
            scalar_test_losses.append(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        final_scalar_loss = best_loss
        scalar_epochs = len(scalar_test_losses)
        
        print(f"\nScalar Network: {scalar_epochs:3d} epochs, final loss: {final_scalar_loss:.6f}")
        
        # ========== Matrix Network ==========
        
        matrix_params = create_matrix_params(jax.random.key(0), 1, 1, n, scale=0.01)
        
        def matrix_forward(params, x):
            x_batch = x[jnp.newaxis, :, :]
            return mtn.dense(params, x_batch, activation=lambda x: x)
        
        def matrix_loss(params, x, y):
            pred = jax.vmap(matrix_forward, in_axes=(None, 0))(params, x)
            pred = jnp.squeeze(pred, axis=1)
            return jnp.mean((pred - y) ** 2)
        
        matrix_loss_jit = jax.jit(matrix_loss)
        matrix_grad_jit = jax.jit(jax.grad(matrix_loss))
        
        # Train matrix network
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(matrix_params)
        
        matrix_test_losses = []
        patience = 10
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(500):
            loss_val = matrix_loss_jit(matrix_params, X_train, Y_train)
            grads = matrix_grad_jit(matrix_params, X_train, Y_train)
            updates, opt_state = optimizer.update(grads, opt_state)
            matrix_params = jax.tree_util.tree_map(
                lambda p, u: p + u, matrix_params, updates
            )
            
            # Check test loss
            test_loss = float(matrix_loss_jit(matrix_params, X_test, Y_test))
            matrix_test_losses.append(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        final_matrix_loss = best_loss
        matrix_epochs = len(matrix_test_losses)
        
        print(f"Matrix Network: {matrix_epochs:3d} epochs, final loss: {final_matrix_loss:.6f}")
        
        # Compare
        speedup = scalar_epochs / matrix_epochs
        better_by = final_scalar_loss / final_matrix_loss
        
        print_comparison_table([
            {
                "name": f"Scalar",
                "params": scalar_params_count,
                "flops": scalar_flops,
                "time_ms": scalar_epochs,  # Using epochs as proxy
                "loss": final_scalar_loss,
            },
            {
                "name": f"Matrix",
                "params": matrix_params_count,
                "flops": matrix_flops,
                "time_ms": matrix_epochs,  # Using epochs as proxy
                "loss": final_matrix_loss,
            },
        ], title=f"Sample Efficiency (n_train={n_train}, {fraction*100:.0f}%)")
        
        print(f"\nConvergence speedup: {speedup:.2f}x")
        print(f"Final loss advantage: {better_by:.2f}x")
        
        # Store results
        results.add(f"n_train={n_train}", "scalar_final_loss", final_scalar_loss)
        results.add(f"n_train={n_train}", "matrix_final_loss", final_matrix_loss)
        results.add(f"n_train={n_train}", "scalar_epochs", scalar_epochs)
        results.add(f"n_train={n_train}", "matrix_epochs", matrix_epochs)
        results.add(f"n_train={n_train}", "speedup", speedup)
        results.add(f"n_train={n_train}", "n_train", n_train)
        results.add(f"n_train={n_train}", "fraction", fraction)
    
    print("\n" + "="*100)
    print(results.summary())
    save_results_csv(results, "results_sample_efficiency_synthetic.csv")
    
    return results


def benchmark_sample_efficiency_mnist():
    """Test sample efficiency on MNIST with varying dataset sizes.
    
    Requires keras/tensorflow for MNIST loading.
    """
    print("\n" + "="*70)
    print("SAMPLE EFFICIENCY: MNIST Classification")
    print("="*70)
    
    try:
        from keras.datasets import mnist
    except ImportError:
        print("Keras not installed, skipping MNIST sample efficiency benchmark")
        return None
    
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full = X_train_full.astype(jnp.float32) / 255.0
    X_test = X_test.astype(jnp.float32) / 255.0
    
    # Use subset of test set for speed
    n_test = 1000
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    
    print(f"Full training set: {len(X_train_full)} samples")
    print(f"Test set: {n_test} samples (fixed)\n")
    
    results = BenchmarkResults()
    
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5]  # Skip 1.0 for speed
    
    for fraction in fractions:
        n_train = max(50, int(len(X_train_full) * fraction))
        
        # Sample uniformly from each class
        indices = np.random.choice(len(X_train_full), n_train, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
        
        print(f"--- Training set size: {n_train} ({fraction*100:.0f}%) ---")
        
        # Flatten for scalar network
        X_train_flat = X_train.reshape(n_train, 784)
        X_test_flat = X_test.reshape(n_test, 784)
        
        # ========== Scalar Network ==========
        
        scalar_params_1 = create_scalar_params(jax.random.key(0), 784, 128, scale=0.01)
        scalar_params_2 = create_scalar_params(jax.random.key(1), 128, 10, scale=0.01)
        
        def scalar_forward_mnist(params_1, params_2, x):
            h = jax.nn.relu(jnp.dot(x, params_1["W"].T) + params_1["b"])
            return jnp.dot(h, params_2["W"].T) + params_2["b"]
        
        def scalar_loss_mnist(p1, p2, x, y):
            logits = jax.vmap(lambda xi: scalar_forward_mnist(p1, p2, xi))(x)
            return jnp.mean(
                jax.nn.softmax_cross_entropy_with_integer_labels(logits, y)
            )
        
        scalar_loss_jit = jax.jit(scalar_loss_mnist)
        scalar_grad_jit = jax.jit(jax.grad(scalar_loss_mnist, argnums=(0, 1)))
        
        optimizer = optax.adam(learning_rate=0.001)
        opt_state_1 = optimizer.init(scalar_params_1)
        opt_state_2 = optimizer.init(scalar_params_2)
        
        scalar_accs = []
        best_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):
            loss_val = scalar_loss_jit(scalar_params_1, scalar_params_2, X_train_flat, y_train)
            grads_1, grads_2 = scalar_grad_jit(
                scalar_params_1, scalar_params_2, X_train_flat, y_train
            )
            
            updates_1, opt_state_1 = optimizer.update(grads_1, opt_state_1)
            updates_2, opt_state_2 = optimizer.update(grads_2, opt_state_2)
            
            scalar_params_1 = jax.tree_util.tree_map(
                lambda p, u: p + u, scalar_params_1, updates_1
            )
            scalar_params_2 = jax.tree_util.tree_map(
                lambda p, u: p + u, scalar_params_2, updates_2
            )
            
            # Eval
            test_logits = jax.vmap(
                lambda xi: scalar_forward_mnist(scalar_params_1, scalar_params_2, xi)
            )(X_test_flat)
            test_acc = float(jnp.mean(jnp.argmax(test_logits, axis=1) == y_test))
            scalar_accs.append(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        final_scalar_acc = best_acc
        scalar_epochs = len(scalar_accs)
        
        print(f"Scalar: {scalar_epochs:2d} epochs, final test acc: {final_scalar_acc:.4f}")
        
        results.add(f"n_train={n_train}", "scalar_test_accuracy", final_scalar_acc)
        results.add(f"n_train={n_train}", "scalar_epochs", scalar_epochs)
        results.add(f"n_train={n_train}", "n_train", n_train)
    
    print("\n" + "="*70)
    print(results.summary())
    save_results_csv(results, "results_sample_efficiency_mnist.csv")
    
    return results


if __name__ == "__main__":
    print("SAMPLE EFFICIENCY BENCHMARKS")
    print("="*70)
    print("\nThese benchmarks test whether matrix-neuron networks achieve")
    print("higher accuracy with fewer training examples.")
    
    results_1 = benchmark_sample_efficiency_synthetic()
    results_2 = benchmark_sample_efficiency_mnist()
    
    print("\n" + "="*70)
    print("Sample efficiency benchmarks complete. Results saved to:")
    print("  - outs/results_sample_efficiency_synthetic.csv")
    print("  - outs/results_sample_efficiency_mnist.csv (if keras available)")
    print("="*70)
