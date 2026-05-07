"""B5 - CIFAR-10 sample efficiency benchmark."""

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


def _to_matrix_tokens(x: jax.Array, n: int, p: int = 48) -> jax.Array:
    flat = x.reshape(x.shape[0], -1)
    token_dim = n * n
    width = p * token_dim
    if width > flat.shape[1]:
        raise ValueError("p*n*n must be <= flattened input size")
    return flat[:, :width].reshape(x.shape[0], p, n, n)


def _train_scalar(
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    y_test: jax.Array,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[float, int, int]:
    params = init_scalar_mlp(jax.random.key(seed), [3072, 2048, 2048, 2048, 2048, 10])
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    @jax.jit
    def loss_fn(p, xb, yb):
        logits = jax.vmap(lambda x: scalar_mlp_forward(p, x))(xb)
        return cross_entropy_from_logits(logits, yb, 10)

    grad_fn = jax.jit(jax.grad(loss_fn))
    lr = 1e-3
    best_acc = 0.0
    patience = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        idx = jax.random.choice(
            jax.random.key(seed + epoch),
            x_train_flat.shape[0],
            (batch_size,),
            replace=False,
        )
        xb, yb = x_train_flat[idx], y_train[idx]
        grads = grad_fn(params, xb, yb)
        lr_val = lr
        params = jax.tree_util.tree_map(
            lambda p, g, lr_=lr_val: p - lr_ * g,
            params,
            grads,
        )
        logits = jax.vmap(lambda x, p=params: scalar_mlp_forward(p, x))(x_test_flat)
        acc = float(
            jnp.mean((jnp.argmax(logits, axis=-1) == y_test).astype(jnp.float32))
        )
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    return best_acc, best_epoch, count_params(params)


def _train_matrix(
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    y_test: jax.Array,
    *,
    n: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[float, int, int]:
    x_train_m = _to_matrix_tokens(x_train, n)
    x_test_m = _to_matrix_tokens(x_test, n)
    p_in = x_train_m.shape[1]
    hidden = 96 if n <= 4 else 64
    matrix_params = init_matrix_mlp(
        jax.random.key(seed), [p_in, hidden, hidden, hidden, hidden], n
    )
    readout = init_scalar_mlp(jax.random.key(seed + 3), [hidden * n * n, 1024, 10])

    @jax.jit
    def logits_fn(mp, rp, xb):
        f = jax.vmap(lambda x: matrix_mlp_forward(mp, x))(xb)
        flat = f.reshape(f.shape[0], -1)
        return jax.vmap(lambda x: scalar_mlp_forward(rp, x))(flat)

    @jax.jit
    def loss_fn(mp, rp, xb, yb):
        return cross_entropy_from_logits(logits_fn(mp, rp, xb), yb, 10)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
    lr = 1e-3
    best_acc = 0.0
    best_epoch = 0
    patience = 0
    for epoch in range(1, epochs + 1):
        idx = jax.random.choice(
            jax.random.key(seed + epoch),
            x_train_m.shape[0],
            (batch_size,),
            replace=False,
        )
        xb, yb = x_train_m[idx], y_train[idx]
        gm, gr = grad_fn(matrix_params, readout, xb, yb)
        lr_val = lr
        matrix_params = jax.tree_util.tree_map(
            lambda p, g, lr_=lr_val: p - lr_ * g,
            matrix_params,
            gm,
        )
        readout = jax.tree_util.tree_map(
            lambda p, g, lr_=lr_val: p - lr_ * g,
            readout,
            gr,
        )
        logits = logits_fn(matrix_params, readout, x_test_m)
        acc = float(
            jnp.mean((jnp.argmax(logits, axis=-1) == y_test).astype(jnp.float32))
        )
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    return best_acc, best_epoch, count_params(matrix_params) + count_params(readout)


def run_b5(
    *,
    epochs: int = 100,
    batch_size: int = 256,
    seed: int = 0,
) -> None:
    x_train, y_train, x_test, y_test = _load_cifar10()
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    rows: list[dict[str, float | int | str]] = []

    for frac in fractions:
        n_train = int(x_train.shape[0] * frac)
        idx = jax.random.choice(
            jax.random.key(seed + int(frac * 1000)),
            x_train.shape[0],
            (n_train,),
            replace=False,
        )
        x_sub, y_sub = x_train[idx], y_train[idx]

        acc_s, epoch_s, p_s = _train_scalar(
            x_sub,
            y_sub,
            x_test,
            y_test,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + int(frac * 10_000),
        )
        rows.append(
            {
                "model": "scalar",
                "fraction": frac,
                "n_train": n_train,
                "test_accuracy": acc_s,
                "best_epoch": epoch_s,
                "params": p_s,
            }
        )

        for n in [2, 4, 8]:
            acc_m, epoch_m, p_m = _train_matrix(
                x_sub,
                y_sub,
                x_test,
                y_test,
                n=n,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed + int(frac * 20_000) + n,
            )
            rows.append(
                {
                    "model": f"matrix_n={n}",
                    "fraction": frac,
                    "n_train": n_train,
                    "test_accuracy": acc_m,
                    "best_epoch": epoch_m,
                    "params": p_m,
                }
            )

    save_rows_csv("examples/benchmarking_v2/outs/results_b5_cifar10_efficiency.csv", rows)
    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b5_accuracy_vs_data.png",
        "B5 CIFAR-10 sample efficiency",
        "training set size",
        "test accuracy",
        {
            series: (
                [float(r["n_train"]) for r in rows if r["model"] == series],
                [float(r["test_accuracy"]) for r in rows if r["model"] == series],
            )
            for series in ["scalar", "matrix_n=2", "matrix_n=4", "matrix_n=8"]
        },
        xscale="log",
    )
    print("B5 complete: outs/results_b5_cifar10_efficiency.csv and plot saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B5 CIFAR-10 sample efficiency benchmark"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b5(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
