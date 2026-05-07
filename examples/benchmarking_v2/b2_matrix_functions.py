"""B2 - Matrix function approximation benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict

import jax
import jax.numpy as jnp

from examples.benchmarking_v2.common import (
    count_params,
    init_matrix_mlp,
    init_scalar_mlp,
    matrix_mlp_forward,
    maybe_plot_lines,
    save_rows_csv,
    scalar_mlp_forward,
)


def _frobenius(pred: jax.Array, target: jax.Array) -> jax.Array:
    diff = pred - target
    return jnp.mean(jnp.sqrt(jnp.sum(diff * diff, axis=(-2, -1))))


def _matrix_targets(x_batch: jax.Array, fn_name: str) -> jax.Array:
    if fn_name == "square":
        return jnp.einsum("bpij,bpjk->bpik", x_batch, x_batch)
    if fn_name == "inverse":
        return jax.vmap(lambda x: jax.vmap(jnp.linalg.inv)(x))(x_batch)
    if fn_name == "exp":
        return jax.vmap(lambda x: jax.vmap(jax.scipy.linalg.expm)(x))(x_batch)
    raise ValueError(f"Unknown function {fn_name}")


def _solve_hidden_scalar(target_params: int, d_in: int, d_out: int) -> int:
    best_h, best_gap = 64, 10**18
    for h in range(32, 4097):
        params = d_in * h + h + h * h + h + h * h + h + h * d_out + d_out
        gap = abs(params - target_params)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def _solve_hidden_matrix(target_params: int, p_in: int, p_out: int, n: int) -> int:
    best_h, best_gap = 8, 10**18
    for h in range(8, 1537):
        params = (
            (h * p_in + h) * n * n
            + (h * h + h) * n * n
            + (h * h + h) * n * n
            + (p_out * h + p_out) * n * n
        )
        gap = abs(params - target_params)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def train_one_function(
    *,
    fn_name: str,
    n: int,
    epochs: int,
    batch_size: int,
    test_size: int,
    target_params: int,
    seed: int,
) -> tuple[list[float], list[float], int, int]:
    key = jax.random.key(seed + n * 13)
    x_train = jax.random.normal(key, (batch_size, 16, n, n)) * 0.2
    x_test = (
        jax.random.normal(jax.random.key(seed + n * 17), (test_size, 16, n, n)) * 0.2
    )
    y_train = _matrix_targets(x_train, fn_name)
    y_test = _matrix_targets(x_test, fn_name)

    h_matrix = _solve_hidden_matrix(target_params, 16, 16, n)
    matrix_params = init_matrix_mlp(
        jax.random.key(seed + n * 23), [16, h_matrix, h_matrix, h_matrix, 16], n
    )

    d = 16 * n * n
    h_scalar = _solve_hidden_scalar(target_params, d, d)
    scalar_params = init_scalar_mlp(
        jax.random.key(seed + n * 29), [d, h_scalar, h_scalar, h_scalar, d]
    )

    @jax.jit
    def matrix_loss(params, xb, yb):
        pred = jax.vmap(lambda x: matrix_mlp_forward(params, x))(xb)
        return _frobenius(pred, yb)

    @jax.jit
    def scalar_loss(params, xb_flat, yb_flat):
        pred = jax.vmap(lambda x: scalar_mlp_forward(params, x))(xb_flat)
        diff = pred - yb_flat
        return jnp.mean(jnp.sqrt(jnp.sum(diff * diff, axis=-1)))

    matrix_grad = jax.jit(jax.grad(matrix_loss))
    scalar_grad = jax.jit(jax.grad(scalar_loss))
    lr = 1e-3

    scalar_curve: list[float] = []
    matrix_curve: list[float] = []
    x_train_flat = x_train.reshape(batch_size, -1)
    y_train_flat = y_train.reshape(batch_size, -1)
    x_test_flat = x_test.reshape(test_size, -1)
    y_test_flat = y_test.reshape(test_size, -1)

    for _ in range(epochs):
        g_matrix = matrix_grad(matrix_params, x_train, y_train)
        matrix_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, matrix_params, g_matrix
        )
        g_scalar = scalar_grad(scalar_params, x_train_flat, y_train_flat)
        scalar_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, scalar_params, g_scalar
        )

        matrix_curve.append(float(matrix_loss(matrix_params, x_test, y_test)))
        scalar_curve.append(float(scalar_loss(scalar_params, x_test_flat, y_test_flat)))

    return (
        scalar_curve,
        matrix_curve,
        count_params(scalar_params),
        count_params(matrix_params),
    )


def run_b2(
    *,
    epochs: int = 500,
    batch_size: int = 256,
    test_size: int = 128,
    target_params: int = 5_000_000,
    seed: int = 0,
) -> None:
    functions = ["square", "inverse", "exp"]
    fn_seed_offset = {"square": 101, "inverse": 211, "exp": 307}
    ns = [2, 4, 8]

    rows: list[dict[str, float | int | str]] = []
    for fn_name in functions:
        scalar_accum = defaultdict(float)
        for n in ns:
            scalar_curve, matrix_curve, scalar_params, matrix_params = (
                train_one_function(
                    fn_name=fn_name,
                    n=n,
                    epochs=epochs,
                    batch_size=batch_size,
                    test_size=test_size,
                    target_params=target_params,
                    seed=seed + fn_seed_offset[fn_name],
                )
            )
            for epoch, (s, m) in enumerate(
                zip(scalar_curve, matrix_curve, strict=True),
                start=1,
            ):
                rows.append(
                    {
                        "task": fn_name,
                        "epoch": epoch,
                        "series": f"matrix_n={n}",
                        "test_frobenius": m,
                        "params": matrix_params,
                    }
                )
                scalar_accum[epoch] += s / len(ns)
        for epoch, value in scalar_accum.items():
            rows.append(
                {
                    "task": fn_name,
                    "epoch": epoch,
                    "series": "scalar",
                    "test_frobenius": value,
                    "params": scalar_params,
                }
            )

        task_rows = [r for r in rows if r["task"] == fn_name]
        maybe_plot_lines(
            f"examples/benchmarking_v2/outs/plot_b2_{fn_name}.png",
            f"B2 {fn_name}: test Frobenius loss",
            "epoch",
            "test Frobenius loss",
            {
                series: (
                    [float(r["epoch"]) for r in task_rows if r["series"] == series],
                    [
                        float(r["test_frobenius"])
                        for r in task_rows
                        if r["series"] == series
                    ],
                )
                for series in ["scalar", "matrix_n=2", "matrix_n=4", "matrix_n=8"]
            },
            yscale="log",
        )

    save_rows_csv("examples/benchmarking_v2/outs/results_b2_matrix_functions.csv", rows)
    print("B2 complete: outs/results_b2_matrix_functions.csv and plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B2 matrix function approximation benchmark"
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--target-params", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        target_params=args.target_params,
        seed=args.seed,
    )
