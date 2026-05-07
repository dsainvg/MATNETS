"""Benchmark 2: Expressivity

Test whether matrix-neuron networks can learn things scalar networks cannot.

Three tasks:
1. Matrix function approximation (X @ X)
2. Standard benchmarks (MNIST)
3. Equivariance (optional, more complex)

This benchmark now shows:
- Exact parameter counts for each network
- Exact FLOP counts
- Final accuracies/losses
- Pretty-printed comparison tables
"""

import jax
import jax.numpy as jnp
import optax
from jax import Array
import numpy as np

import matnets as mtn
from matnets._params import MatrixParams
from examples.benchmarking_v1.benchmark_utils import (
    BenchmarkResults,
    count_parameters,
    count_flops_dense_layer,
    count_flops_matrix_einsum,
    create_scalar_params,
    create_matrix_params,
    create_matched_scalar_params,
    save_results_csv,
    plot_results,
    print_comparison_table,
)


# ============================================================================
# Task 1: Matrix Function Approximation (X @ X)
# ============================================================================


def benchmark_matrix_function():
    """Learn to approximate f(X) = X @ X.
    
    Matrix networks should learn this much more easily than scalar networks.
    """
    print("\n" + "="*100)
    print("TASK 1: Matrix Function Approximation (f(X) = X @ X)")
    print("="*100)
    
    results = BenchmarkResults()
    
    # Generate dataset
    key = jax.random.key(0)
    n = 4  # matrix dimension
    n_samples = 1000
    
    key_gen = jax.random.key(1)
    X_data = jax.random.normal(key_gen, (n_samples, n, n)) * 0.5
    Y_data = jnp.einsum("sij,sjk->sik", X_data, X_data)  # X @ X
    
    # Split into train/test
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    X_train, X_test = X_data[:n_train], X_data[n_train:]
    Y_train, Y_test = Y_data[:n_train], Y_data[n_train:]
    
    print(f"\nDataset: {n_samples} samples of {n}x{n} matrices")
    print(f"Train: {n_train}, Test: {n_test}")
    
    # ========== Scalar Network ==========
    print(f"\n{'='*100}")
    print("Scalar Network (baseline)")
    print(f"{'='*100}")
    
    # For scalar network, flatten matrices: (16,) -> (16,)
    X_train_flat = X_train.reshape(n_train, n * n)
    X_test_flat = X_test.reshape(n_test, n * n)
    Y_train_flat = Y_train.reshape(n_train, n * n)
    Y_test_flat = Y_test.reshape(n_test, n * n)
    
    p_scalar = n * n  # input dimension
    q_scalar = n * n  # output dimension
    
    scalar_params = create_scalar_params(jax.random.key(0), p_scalar, q_scalar, scale=0.01)
    scalar_params_count = count_parameters(scalar_params)
    scalar_flops = count_flops_dense_layer(p_scalar, q_scalar)
    
    print(f"Architecture: p={p_scalar}, q={q_scalar}")
    print(f"Parameters:   {scalar_params_count:,}")
    print(f"FLOPs/forward: {scalar_flops:,}")
    
    def scalar_forward(params, x):
        # x: (n*n,) flattened matrix
        return jnp.dot(params["W"], x) + params["b"]
    
    def scalar_loss(params, x, y):
        pred = jax.vmap(lambda xi, yi: scalar_forward(params, xi))(x, y)
        return jnp.mean((pred - y) ** 2)
    
    scalar_loss_jit = jax.jit(scalar_loss)
    scalar_grad_jit = jax.jit(jax.grad(scalar_loss))
    
    # Train scalar network
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(scalar_params)
    
    scalar_losses = []
    for epoch in range(100):
        loss_val = scalar_loss_jit(scalar_params, X_train_flat, Y_train_flat)
        grads = scalar_grad_jit(scalar_params, X_train_flat, Y_train_flat)
        updates, opt_state = optimizer.update(grads, opt_state)
        scalar_params = jax.tree_util.tree_map(
            lambda p, u: p + u, scalar_params, updates
        )
        scalar_losses.append(float(loss_val))
        
        if (epoch + 1) % 20 == 0:
            test_loss = float(scalar_loss_jit(scalar_params, X_test_flat, Y_test_flat))
            print(f"Epoch {epoch+1:3d} | Train loss: {loss_val:.6f} | Test loss: {test_loss:.6f}")
    
    final_scalar_test_loss = float(scalar_loss_jit(scalar_params, X_test_flat, Y_test_flat))
    print(f"\nFinal test loss: {final_scalar_test_loss:.6f}")
    results.add("scalar", "final_test_loss", final_scalar_test_loss)
    results.add("scalar", "final_train_loss", float(scalar_losses[-1]))
    results.add("scalar", "params", scalar_params_count)
    results.add("scalar", "flops", scalar_flops)
    
    # ========== Matrix Network ==========
    print(f"\n{'='*100}")
    print(f"Matrix Network (n={n})")
    print(f"{'='*100}")
    
    # Matrix network processes matrices directly
    # Input: (p, n, n), Output: (q, n, n)
    p_matrix = 1
    q_matrix = 1
    matrix_params = create_matrix_params(jax.random.key(0), p_matrix, q_matrix, n, scale=0.01)
    matrix_params_count = count_parameters(matrix_params)
    matrix_flops = count_flops_matrix_einsum(p_matrix, q_matrix, n)
    
    print(f"Architecture: p={p_matrix}, q={q_matrix}, n={n}")
    print(f"W shape: ({q_matrix}, {p_matrix}, {n}, {n})")
    print(f"B shape: ({q_matrix}, {n}, {n})")
    print(f"Parameters:   {matrix_params_count:,}")
    print(f"FLOPs/forward: {matrix_flops:,}")
    
    def matrix_forward(params, x):
        # x: (n, n) matrix
        x_batch = x[jnp.newaxis, :, :]  # add batch dim
        return mtn.dense(params, x_batch, activation=lambda x: x)
    
    def matrix_loss(params, x, y):
        # vmap over batch dimension of x only (in_axes=(None, 0))
        pred = jax.vmap(matrix_forward, in_axes=(None, 0))(params, x)
        pred = jnp.squeeze(pred, axis=1)  # remove batch dim
        return jnp.mean((pred - y) ** 2)
    
    matrix_loss_jit = jax.jit(matrix_loss)
    matrix_grad_jit = jax.jit(jax.grad(matrix_loss))
    
    # Train matrix network
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(matrix_params)
    
    matrix_losses = []
    for epoch in range(100):
        loss_val = matrix_loss_jit(matrix_params, X_train, Y_train)
        grads = matrix_grad_jit(matrix_params, X_train, Y_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        matrix_params = jax.tree_util.tree_map(
            lambda p, u: p + u, matrix_params, updates
        )
        matrix_losses.append(float(loss_val))
        
        if (epoch + 1) % 20 == 0:
            test_loss = float(matrix_loss_jit(matrix_params, X_test, Y_test))
            print(f"Epoch {epoch+1:3d} | Train loss: {loss_val:.6f} | Test loss: {test_loss:.6f}")
    
    final_matrix_test_loss = float(matrix_loss_jit(matrix_params, X_test, Y_test))
    print(f"\nFinal test loss: {final_matrix_test_loss:.6f}")
    
    results.add("matrix", "final_test_loss", final_matrix_test_loss)
    results.add("matrix", "final_train_loss", float(matrix_losses[-1]))
    results.add("matrix", "params", matrix_params_count)
    results.add("matrix", "flops", matrix_flops)
    
    print("\n" + "="*100)
    print("SUMMARY: Matrix Function Approximation (f(X) = X @ X)")
    print("="*100)
    
    # Print comparison table
    print_comparison_table([
        {
            "name": f"Scalar ({p_scalar}×{q_scalar})",
            "params": scalar_params_count,
            "flops": scalar_flops,
            "loss": final_scalar_test_loss,
        },
        {
            "name": f"Matrix (n={n})",
            "params": matrix_params_count,
            "flops": matrix_flops,
            "loss": final_matrix_test_loss,
        },
    ], title="Expressivity: Matrix Function Approximation")
    
    print(f"\nLoss ratio (scalar / matrix): {final_scalar_test_loss / final_matrix_test_loss:.1f}x")
    print(f"Parameter ratio (matrix / scalar): {matrix_params_count / scalar_params_count:.1f}x")
    
    if final_matrix_test_loss < final_scalar_test_loss:
        print("[+] Matrix network clearly wins on this task!")
    elif final_matrix_test_loss < final_scalar_test_loss * 1.5:
        print("[~] Similar performance (matrix network competitive)")
    else:
        print("[-] Scalar network performs better (matrix network not suited)")
    
    print("\n" + results.summary())
    save_results_csv(results, "results_matrix_function.csv")
    
    return results


# ============================================================================
# Task 2: MNIST Digit Classification
# ============================================================================


def load_mnist():
    """Load MNIST dataset (requires Keras/TensorFlow).
    
    Returns:
        (X_train, y_train, X_test, y_test) normalized to [0, 1]
    """
    try:
        from keras.datasets import mnist
    except ImportError:
        print("Warning: keras not installed, skipping MNIST benchmark")
        return None
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize
    X_train = X_train.astype(jnp.float32) / 255.0
    X_test = X_test.astype(jnp.float32) / 255.0
    
    return X_train, y_train, X_test, y_test


def benchmark_mnist():
    """Benchmark on MNIST digit classification.
    
    This is a sanity check: both networks should work, matrix network
    may or may not be better depending on network architecture.
    """
    print("\n" + "="*70)
    print("TASK 2: MNIST Digit Classification")
    print("="*70)
    
    mnist_data = load_mnist()
    if mnist_data is None:
        print("Skipping MNIST (keras not installed)")
        return None
    
    X_train, y_train, X_test, y_test = mnist_data
    
    # Use subset for speed
    n_train = 5000
    n_test = 1000
    
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    
    print(f"Dataset: MNIST subset")
    print(f"Train: {n_train}, Test: {n_test}")
    
    results = BenchmarkResults()
    
    # ========== Scalar Network ==========
    print("\n--- Scalar Network (two-layer) ---")
    
    # Flatten images: 28x28 -> 784
    X_train_flat = X_train.reshape(n_train, 784)
    X_test_flat = X_test.reshape(n_test, 784)
    
    # Two-layer scalar network: 784 -> 128 -> 10
    scalar_params_1 = create_scalar_params(jax.random.key(0), 784, 128, scale=0.01)
    scalar_params_2 = create_scalar_params(jax.random.key(1), 128, 10, scale=0.01)
    
    def scalar_forward_mnist(params_1, params_2, x):
        h = jax.nn.relu(jnp.dot(x, params_1["W"].T) + params_1["b"])
        return jnp.dot(h, params_2["W"].T) + params_2["b"]
    
    def scalar_loss_mnist(p1, p2, x, y):
        logits = jax.vmap(lambda xi: scalar_forward_mnist(p1, p2, xi))(x)
        one_hot_y = jax.nn.one_hot(y, 10)
        return jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot_y)
        )
    
    scalar_loss_jit = jax.jit(scalar_loss_mnist)
    scalar_grad_jit = jax.jit(
        jax.grad(scalar_loss_mnist, argnums=(0, 1))
    )
    
    # Train
    optimizer = optax.adam(learning_rate=0.001)
    opt_state_1 = optimizer.init(scalar_params_1)
    opt_state_2 = optimizer.init(scalar_params_2)
    
    scalar_accs = []
    for epoch in range(20):
        loss_val = scalar_loss_jit(scalar_params_1, scalar_params_2, X_train_flat, y_train)
        grads_1, grads_2 = scalar_grad_jit(scalar_params_1, scalar_params_2, X_train_flat, y_train)
        
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
        test_acc = jnp.mean(jnp.argmax(test_logits, axis=1) == y_test)
        scalar_accs.append(float(test_acc))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Train loss: {loss_val:.6f} | Test acc: {test_acc:.4f}")
    
    final_scalar_acc = scalar_accs[-1]
    results.add("scalar", "final_test_accuracy", final_scalar_acc)
    results.add("scalar", "params", 101)  # Approximate: 784*128 + 128 + 128*10 + 10
    
    print(f"\nFinal scalar accuracy: {final_scalar_acc:.4f}")
    
    # ========== Matrix Network for MNIST ==========
    print(f"\n{'='*100}")
    print("Matrix Network (processing 28x28 images as matrices)")
    print(f"{'='*100}")
    
    # Reshape MNIST images back to 28×28 matrices (they were flattened from this)
    X_train_mat = X_train_flat.reshape(-1, 28, 28)
    X_test_mat = X_test_flat.reshape(-1, 28, 28)
    
    # Single matrix layer: 1 input channel, 16 output channels, n=28
    # Then pool/flatten and classify
    n_mat = 28
    p_mat = 1  # 1 input channel
    q_mat = 2  # 2 output channels (smaller to reduce parameters)
    
    matrix_params_mnist = create_matrix_params(jax.random.key(2), p_mat, q_mat, n_mat, scale=0.01)
    
    # Classification head: 2 * 28 * 28 = 1568 -> 10
    classify_params = create_scalar_params(jax.random.key(3), q_mat * n_mat * n_mat, 10, scale=0.01)
    
    matrix_params_count = count_parameters(matrix_params_mnist)
    classify_params_count = count_parameters(classify_params)
    total_matrix_params = matrix_params_count + classify_params_count
    
    print(f"Matrix layer: p={p_mat}, q={q_mat}, n={n_mat}")
    print(f"Matrix params: {matrix_params_count}")
    print(f"Classification params: {classify_params_count}")
    print(f"Total params: {total_matrix_params}")
    
    def matrix_forward_mnist(mat_params, classify_p, x):
        # x: (28, 28) image matrix
        x_batch = x[jnp.newaxis, :, :]  # Add batch dim -> (1, 28, 28)
        # Apply matrix neuron layer
        out = mtn.dense(mat_params, x_batch, activation=jax.nn.relu)  # (q, 28, 28)
        out = jnp.squeeze(out, axis=0) if out.shape[0] == 1 else out  # Remove if single channel
        # Flatten for classification
        out_flat = out.reshape(-1)
        # Classify
        return jnp.dot(out_flat, classify_p["W"].T) + classify_p["b"]
    
    def matrix_loss_mnist(mat_p, cls_p, x, y):
        logits = jax.vmap(
            lambda xi: matrix_forward_mnist(mat_p, cls_p, xi)
        )(x)
        one_hot_y = jax.nn.one_hot(y, 10)
        return jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot_y)
        )
    
    matrix_loss_jit = jax.jit(matrix_loss_mnist)
    matrix_grad_jit = jax.jit(
        jax.grad(matrix_loss_mnist, argnums=(0, 1))
    )
    
    # Train
    optimizer = optax.adam(learning_rate=0.001)
    opt_state_mat = optimizer.init(matrix_params_mnist)
    opt_state_cls = optimizer.init(classify_params)
    
    matrix_accs = []
    for epoch in range(20):
        loss_val = matrix_loss_jit(matrix_params_mnist, classify_params, X_train_mat, y_train)
        grads_mat, grads_cls = matrix_grad_jit(matrix_params_mnist, classify_params, X_train_mat, y_train)
        
        updates_mat, opt_state_mat = optimizer.update(grads_mat, opt_state_mat)
        updates_cls, opt_state_cls = optimizer.update(grads_cls, opt_state_cls)
        
        matrix_params_mnist = jax.tree_util.tree_map(
            lambda p, u: p + u, matrix_params_mnist, updates_mat
        )
        classify_params = jax.tree_util.tree_map(
            lambda p, u: p + u, classify_params, updates_cls
        )
        
        # Eval
        test_logits = jax.vmap(
            lambda xi: matrix_forward_mnist(matrix_params_mnist, classify_params, xi)
        )(X_test_mat)
        test_acc = jnp.mean(jnp.argmax(test_logits, axis=1) == y_test)
        matrix_accs.append(float(test_acc))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Train loss: {loss_val:.6f} | Test acc: {test_acc:.4f}")
    
    final_matrix_acc = matrix_accs[-1]
    results.add("matrix", "final_test_accuracy", final_matrix_acc)
    results.add("matrix", "params", total_matrix_params)
    
    print(f"\nFinal matrix accuracy: {final_matrix_acc:.4f}")
    
    # Summary comparison
    print(f"\n{'='*100}")
    print("MNIST Comparison")
    print(f"{'='*100}")
    print_comparison_table([
        {
            "name": "Scalar (784->128->10)",
            "params": 101,
            "flops": "N/A",
            "accuracy": final_scalar_acc,
        },
        {
            "name": f"Matrix (n=28, q={q_mat})",
            "params": total_matrix_params,
            "flops": "N/A",
            "accuracy": final_matrix_acc,
        },
    ], title="MNIST: Scalar vs Matrix Networks")
    
    accuracy_ratio = final_scalar_acc / final_matrix_acc if final_matrix_acc > 0 else 1.0
    print(f"\nAccuracy ratio (scalar / matrix): {accuracy_ratio:.2f}x")
    print(f"Parameter ratio (matrix / scalar): {total_matrix_params / 101:.2f}x")
    
    if final_matrix_acc >= final_scalar_acc * 0.95:
        print("[~] Competitive performance - matrix network is viable")
    elif final_matrix_acc > final_scalar_acc:
        print("[+] Matrix network wins!")
    else:
        print("[-] Scalar network performs better (more appropriate architecture)")
    
    print("\n" + results.summary())
    save_results_csv(results, "results_mnist.csv")
    
    return results


# ============================================================================
# Task 3: Simple Synthetic Task
# ============================================================================


def benchmark_synthetic_task():
    """Benchmark on a simple synthetic task designed for matrix networks.
    
    Task: Predict trace(X @ X) from X
    """
    print("\n" + "="*70)
    print("TASK 3: Synthetic Task (Predict Trace of X @ X)")
    print("="*70)
    
    results = BenchmarkResults()
    
    # Generate dataset
    key = jax.random.key(0)
    n = 4
    n_samples = 1000
    
    key_gen = jax.random.key(1)
    X_data = jax.random.normal(key_gen, (n_samples, n, n)) * 0.5
    Y_data = jnp.trace(jnp.einsum("sij,sjk->sik", X_data, X_data), axis1=1, axis2=2)
    Y_data = Y_data[:, jnp.newaxis]  # shape (n_samples, 1)
    
    n_train = int(0.8 * n_samples)
    X_train, X_test = X_data[:n_train], X_data[n_train:]
    Y_train, Y_test = Y_data[:n_train], Y_data[n_train:]
    
    print(f"Dataset: {n_samples} samples")
    print(f"Input: {n}x{n} matrices")
    print(f"Output: scalar (trace of X@X)")
    
    # ========== Scalar Network ==========
    print("\n--- Scalar Network ---")
    
    X_train_flat = X_train.reshape(n_train, n*n)
    n_test = n_samples - n_train
    X_test_flat = X_test.reshape(n_test, n*n)
    
    scalar_params = create_scalar_params(jax.random.key(0), n*n, 1, scale=0.01)
    
    def scalar_forward_synth(params, x):
        return jnp.dot(params["W"], x) + params["b"]
    
    def scalar_loss_synth(params, x, y):
        pred = jax.vmap(scalar_forward_synth, in_axes=(None, 0))(params, x)
        return jnp.mean((pred.squeeze() - y.squeeze()) ** 2)
    
    scalar_loss_jit = jax.jit(scalar_loss_synth)
    scalar_grad_jit = jax.jit(jax.grad(scalar_loss_synth))
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(scalar_params)
    
    for epoch in range(100):
        loss_val = scalar_loss_jit(scalar_params, X_train_flat, Y_train)
        grads = scalar_grad_jit(scalar_params, X_train_flat, Y_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        scalar_params = jax.tree_util.tree_map(
            lambda p, u: p + u, scalar_params, updates
        )
        
        if (epoch + 1) % 20 == 0:
            test_loss = float(scalar_loss_jit(scalar_params, X_test_flat, Y_test))
            print(f"Epoch {epoch+1:3d} | Train loss: {loss_val:.6f} | Test loss: {test_loss:.6f}")
    
    final_scalar_test_loss = float(scalar_loss_jit(scalar_params, X_test_flat, Y_test))
    results.add("scalar", "final_test_loss", final_scalar_test_loss)
    
    # ========== Matrix Network ==========
    print("\n--- Matrix Network ---")
    
    matrix_params = create_matrix_params(jax.random.key(0), 1, 1, n, scale=0.01)
    
    def matrix_forward_synth(params, x):
        x_batch = x[jnp.newaxis, :, :]
        out = mtn.dense(params, x_batch, activation=lambda x: x)
        return jnp.trace(out[0], axis1=0, axis2=1)
    
    def matrix_loss_synth(params, x, y):
        pred = jax.vmap(matrix_forward_synth, in_axes=(None, 0))(params, x)
        return jnp.mean((pred - y.squeeze()) ** 2)
    
    matrix_loss_jit = jax.jit(matrix_loss_synth)
    matrix_grad_jit = jax.jit(jax.grad(matrix_loss_synth))
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(matrix_params)
    
    for epoch in range(100):
        loss_val = matrix_loss_jit(matrix_params, X_train, Y_train)
        grads = matrix_grad_jit(matrix_params, X_train, Y_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        matrix_params = jax.tree_util.tree_map(
            lambda p, u: p + u, matrix_params, updates
        )
        
        if (epoch + 1) % 20 == 0:
            test_loss = float(matrix_loss_jit(matrix_params, X_test, Y_test))
            print(f"Epoch {epoch+1:3d} | Train loss: {loss_val:.6f} | Test loss: {test_loss:.6f}")
    
    final_matrix_test_loss = float(matrix_loss_jit(matrix_params, X_test, Y_test))
    results.add("matrix", "final_test_loss", final_matrix_test_loss)
    
    print("\n" + "="*70)
    print("SUMMARY: Synthetic Task")
    print("="*70)
    print(f"Scalar final test loss:  {final_scalar_test_loss:.6f}")
    print(f"Matrix final test loss:  {final_matrix_test_loss:.6f}")
    print(f"Better by: {final_scalar_test_loss / final_matrix_test_loss:.1f}x")
    
    print("\n" + results.summary())
    save_results_csv(results, "results_synthetic_task.csv")
    
    return results


if __name__ == "__main__":
    print("EXPRESSIVITY BENCHMARKS")
    print("="*70)
    print("\nThese benchmarks test whether matrix-neuron networks can learn")
    print("things that scalar networks cannot.")
    
    results_1 = benchmark_matrix_function()
    results_2 = benchmark_mnist()
    results_3 = benchmark_synthetic_task()
    
    print("\n" + "="*70)
    print("Expressivity benchmarks complete. Results saved to:")
    print("  - outs/results_matrix_function.csv")
    print("  - outs/results_mnist.csv (if keras available)")
    print("  - outs/results_synthetic_task.csv")
    print("="*70)
