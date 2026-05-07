"""B3 - Symmetry / Equivariance generalization benchmark."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from examples.benchmarking_v2.common import (
    count_params,
    init_matrix_mlp,
    init_scalar_mlp,
    matrix_mlp_forward,
    save_rows_csv,
    scalar_mlp_forward,
)


def _random_rotation(key: jax.Array, n: int) -> jax.Array:
    q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    return q


def _target_function(x_batch: jax.Array) -> jax.Array:
    return jnp.einsum("bpij,bpjk->bpik", x_batch, x_batch)


def run_b3(
    *,
    n: int = 8,
    n_train: int = 50_000,
    n_test: int = 2_000,
    batch_size: int = 256,
    epochs: int = 100,
    target_params: int = 10_000_000,
    seed: int = 0,
) -> None:
    key = jax.random.key(seed)
    x_train = jax.random.normal(key, (n_train, 16, n, n)) * 0.15
    y_train = _target_function(x_train)
    x_test = jax.random.normal(jax.random.key(seed + 1), (n_test, 16, n, n)) * 0.15
    r = _random_rotation(jax.random.key(seed + 2), n)
    x_test_rot = jnp.einsum("ai,bpij,jc->bpac", r, x_test, r.T)
    y_test_rot = _target_function(x_test_rot)

    h_matrix = max(32, int(((target_params // (n * n)) // 2) ** 0.5))
    matrix_params = init_matrix_mlp(
        jax.random.key(seed + 3), [16, h_matrix, h_matrix, h_matrix, 16], n
    )

    d = 16 * n * n
    h_scalar = max(64, int((target_params // 2) ** 0.5))
    scalar_params = init_scalar_mlp(
        jax.random.key(seed + 4), [d, h_scalar, h_scalar, h_scalar, d]
    )

    @jax.jit
    def matrix_loss(params, xb, yb):
        pred = jax.vmap(lambda x: matrix_mlp_forward(params, x))(xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def scalar_loss(params, xb, yb):
        pred = jax.vmap(lambda x: scalar_mlp_forward(params, x))(xb)
        return jnp.mean((pred - yb) ** 2)

    matrix_grad = jax.jit(jax.grad(matrix_loss))
    scalar_grad = jax.jit(jax.grad(scalar_loss))
    lr = 1e-3

    for epoch in range(1, epochs + 1):
        idx = jax.random.choice(
            jax.random.key(seed + 10 + epoch),
            n_train,
            (batch_size,),
            replace=False,
        )
        xb = x_train[idx]
        yb = y_train[idx]
        g_matrix = matrix_grad(matrix_params, xb, yb)
        matrix_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, matrix_params, g_matrix
        )
        xb_flat = xb.reshape(batch_size, -1)
        yb_flat = yb.reshape(batch_size, -1)
        g_scalar = scalar_grad(scalar_params, xb_flat, yb_flat)
        scalar_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, scalar_params, g_scalar
        )

    matrix_test_rot_loss = float(matrix_loss(matrix_params, x_test_rot, y_test_rot))
    scalar_test_rot_loss = float(
        scalar_loss(
            scalar_params,
            x_test_rot.reshape(n_test, -1),
            y_test_rot.reshape(n_test, -1),
        )
    )

    rows = [
        {
            "model": "scalar",
            "n": n,
            "test_rotated_loss": scalar_test_rot_loss,
            "params": count_params(scalar_params),
        },
        {
            "model": "matrix",
            "n": n,
            "test_rotated_loss": matrix_test_rot_loss,
            "params": count_params(matrix_params),
        },
    ]
    save_rows_csv("examples/benchmarking_v2/outs/results_b3_equivariance.csv", rows)
    print("B3 complete: outs/results_b3_equivariance.csv saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B3 equivariance benchmark")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--target-params", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b3(
        n=args.n,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        target_params=args.target_params,
        seed=args.seed,
    )
