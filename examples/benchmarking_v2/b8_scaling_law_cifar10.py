"""B8 - Scaling law experiment on CIFAR-10."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from examples.benchmarking_v2.common import (
    count_params,
    cross_entropy_from_logits,
    init_matrix_mlp,
    init_scalar_mlp,
    matrix_mlp_forward,
    maybe_plot_lines,
    save_rows_csv,
    scalar_mlp_forward,
)


def _load_cifar10() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    try:
        from keras.datasets import cifar10
    except ImportError as exc:
        raise RuntimeError("Keras/TensorFlow required for CIFAR-10 benchmark.") from exc
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = jnp.asarray(x_train, dtype=jnp.float32) / 255.0
    x_test = jnp.asarray(x_test, dtype=jnp.float32) / 255.0
    return (
        x_train,
        jnp.asarray(y_train).reshape(-1),
        x_test,
        jnp.asarray(y_test).reshape(-1),
    )


def _scalar_hidden_for_budget(p: int, q: int, budget: int) -> int:
    best_h, best_gap = 64, 10**18
    for h in range(16, 4097):
        params = p * h + h + h * h + h + h * h + h + h * h + h + h * q + q
        gap = abs(params - budget)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def _matrix_hidden_for_budget(p: int, q: int, n: int, budget: int) -> int:
    best_h, best_gap = 8, 10**18
    for h in range(8, 1537):
        params = (
            (h * p + h) * n * n
            + (h * h + h) * n * n
            + (h * h + h) * n * n
            + (h * h + h) * n * n
            + (q * h + q) * n * n
        )
        gap = abs(params - budget)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def run_b8(
    *,
    epochs: int = 80,
    batch_size: int = 256,
    n: int = 4,
    seed: int = 0,
) -> None:
    budgets = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]
    x_train, y_train, x_test, y_test = _load_cifar10()
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    p_scalar = x_train_flat.shape[1]
    q = 10

    rows: list[dict[str, float | int | str]] = []

    for budget in budgets:
        h_s = _scalar_hidden_for_budget(p_scalar, q, budget)
        scalar = init_scalar_mlp(
            jax.random.key(seed + budget), [p_scalar, h_s, h_s, h_s, h_s, q]
        )

        @jax.jit
        def scalar_loss(params, xb, yb):
            logits = jax.vmap(lambda x: scalar_mlp_forward(params, x))(xb)
            return cross_entropy_from_logits(logits, yb, q)

        g_scalar = jax.jit(jax.grad(scalar_loss))
        lr = 1e-3
        for epoch in range(1, epochs + 1):
            idx = jax.random.choice(
                jax.random.key(seed + budget + epoch),
                x_train_flat.shape[0],
                (batch_size,),
                replace=False,
            )
            xb, yb = x_train_flat[idx], y_train[idx]
            grads = g_scalar(scalar, xb, yb)
            lr_val = lr
            scalar = jax.tree_util.tree_map(
                lambda p, g, lr_=lr_val: p - lr_ * g,
                scalar,
                grads,
            )
        logits = jax.vmap(lambda x, p=scalar: scalar_mlp_forward(p, x))(x_test_flat)
        acc_s = float(
            jnp.mean((jnp.argmax(logits, axis=-1) == y_test).astype(jnp.float32))
        )
        rows.append(
            {
                "model": "scalar",
                "budget": budget,
                "test_accuracy": acc_s,
                "params": count_params(scalar),
            }
        )

        p_m = 48
        x_train_m = x_train_flat[:, : p_m * n * n].reshape(x_train.shape[0], p_m, n, n)
        x_test_m = x_test_flat[:, : p_m * n * n].reshape(x_test.shape[0], p_m, n, n)
        h_m = _matrix_hidden_for_budget(p_m, 1, n, budget)
        matrix = init_matrix_mlp(
            jax.random.key(seed + budget + 3), [p_m, h_m, h_m, h_m, h_m, 1], n
        )
        readout = init_scalar_mlp(jax.random.key(seed + budget + 4), [n * n, 128, q])

        @jax.jit
        def matrix_logits(mp, rp, xb):
            feats = jax.vmap(lambda x: matrix_mlp_forward(mp, x))(xb)
            flat = feats.reshape(feats.shape[0], -1)
            return jax.vmap(lambda x: scalar_mlp_forward(rp, x))(flat)

        @jax.jit
        def matrix_loss(mp, rp, xb, yb):
            return cross_entropy_from_logits(matrix_logits(mp, rp, xb), yb, q)

        g_matrix = jax.jit(jax.grad(matrix_loss, argnums=(0, 1)))
        for epoch in range(1, epochs + 1):
            idx = jax.random.choice(
                jax.random.key(seed + budget + 100 + epoch),
                x_train_m.shape[0],
                (batch_size,),
                replace=False,
            )
            xb, yb = x_train_m[idx], y_train[idx]
            gm, gr = g_matrix(matrix, readout, xb, yb)
            lr_val = lr
            matrix = jax.tree_util.tree_map(
                lambda p, g, lr_=lr_val: p - lr_ * g,
                matrix,
                gm,
            )
            readout = jax.tree_util.tree_map(
                lambda p, g, lr_=lr_val: p - lr_ * g,
                readout,
                gr,
            )
        logits = matrix_logits(matrix, readout, x_test_m)
        acc_m = float(
            jnp.mean((jnp.argmax(logits, axis=-1) == y_test).astype(jnp.float32))
        )
        rows.append(
            {
                "model": "matrix_n=4",
                "budget": budget,
                "test_accuracy": acc_m,
                "params": count_params(matrix) + count_params(readout),
            }
        )

    save_rows_csv("examples/benchmarking_v2/outs/results_b8_scaling_law.csv", rows)
    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b8_scaling_law.png",
        "B8 Scaling law on CIFAR-10",
        "parameter count",
        "test accuracy",
        {
            series: (
                [float(r["params"]) for r in rows if r["model"] == series],
                [float(r["test_accuracy"]) for r in rows if r["model"] == series],
            )
            for series in ["scalar", "matrix_n=4"]
        },
        xscale="log",
    )
    print("B8 complete: outs/results_b8_scaling_law.csv and plot saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B8 scaling law benchmark")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b8(epochs=args.epochs, batch_size=args.batch_size, n=args.n, seed=args.seed)
