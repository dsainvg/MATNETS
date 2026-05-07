"""B4 - MNIST at scale benchmark."""

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


def _load_mnist() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    try:
        from keras.datasets import mnist
    except ImportError as exc:
        raise RuntimeError("Keras/TensorFlow required for MNIST benchmark.") from exc
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = jnp.asarray(x_train, dtype=jnp.float32) / 255.0
    x_test = jnp.asarray(x_test, dtype=jnp.float32) / 255.0
    return x_train, jnp.asarray(y_train), x_test, jnp.asarray(y_test)


def _mnist_to_matrix_tokens(x: jax.Array, n: int) -> jax.Array:
    d = 784
    token_dim = n * n
    p = max(1, d // token_dim)
    x_flat = x.reshape(x.shape[0], d)
    x_trim = x_flat[:, : p * token_dim]
    return x_trim.reshape(x.shape[0], p, n, n)


def run_b4(
    *,
    epochs: int = 100,
    batch_size: int = 512,
    seed: int = 0,
) -> None:
    x_train, y_train, x_test, y_test = _load_mnist()
    ns = [2, 4, 8]
    rows: list[dict[str, float | int | str]] = []
    scalar_curve: list[float] = []

    scalar_dims = [784, 1024, 1024, 1024, 1024, 10]
    scalar_params = init_scalar_mlp(jax.random.key(seed), scalar_dims)

    @jax.jit
    def scalar_loss(params, xb, yb):
        logits = jax.vmap(lambda x: scalar_mlp_forward(params, x))(xb)
        return cross_entropy_from_logits(logits, yb, 10)

    scalar_grad = jax.jit(jax.grad(scalar_loss))
    lr = 1e-3
    x_train_flat = x_train.reshape(x_train.shape[0], 784)
    x_test_flat = x_test.reshape(x_test.shape[0], 784)

    for epoch in range(1, epochs + 1):
        idx = jax.random.choice(
            jax.random.key(seed + epoch), x_train.shape[0], (batch_size,), replace=False
        )
        xb, yb = x_train_flat[idx], y_train[idx]
        grads = scalar_grad(scalar_params, xb, yb)
        lr_val = lr
        scalar_params = jax.tree_util.tree_map(
            lambda p, g, lr_=lr_val: p - lr_ * g, scalar_params, grads
        )
        test_logits = jax.vmap(
            lambda x, p=scalar_params: scalar_mlp_forward(p, x)
        )(x_test_flat)
        acc = float(
            jnp.mean((jnp.argmax(test_logits, axis=-1) == y_test).astype(jnp.float32))
        )
        scalar_curve.append(acc)

    for n in ns:
        x_train_m = _mnist_to_matrix_tokens(x_train, n)
        x_test_m = _mnist_to_matrix_tokens(x_test, n)
        p_in = x_train_m.shape[1]
        hidden = 128
        proj_params = init_matrix_mlp(
            jax.random.key(seed + n * 5),
            [p_in, hidden, hidden, hidden, hidden],
            n,
        )
        readout = init_scalar_mlp(
            jax.random.key(seed + n * 7), [hidden * n * n, 512, 10]
        )

        @jax.jit
        def matrix_logits(m_params, ro_params, xb):
            features = jax.vmap(lambda x: matrix_mlp_forward(m_params, x))(xb)
            flat = features.reshape(features.shape[0], -1)
            return jax.vmap(lambda v: scalar_mlp_forward(ro_params, v))(flat)

        @jax.jit
        def matrix_loss(m_params, ro_params, xb, yb):
            return cross_entropy_from_logits(
                matrix_logits(m_params, ro_params, xb), yb, 10
            )

        grad_m = jax.jit(jax.grad(matrix_loss, argnums=(0, 1)))
        curve: list[float] = []
        for epoch in range(1, epochs + 1):
            idx = jax.random.choice(
                jax.random.key(seed + n * 100 + epoch),
                x_train_m.shape[0],
                (batch_size,),
                replace=False,
            )
            xb, yb = x_train_m[idx], y_train[idx]
            gm, gr = grad_m(proj_params, readout, xb, yb)
            lr_val = lr
            proj_params = jax.tree_util.tree_map(
                lambda p, g, lr_=lr_val: p - lr_ * g, proj_params, gm
            )
            readout = jax.tree_util.tree_map(
                lambda p, g, lr_=lr_val: p - lr_ * g,
                readout,
                gr,
            )
            logits = matrix_logits(proj_params, readout, x_test_m)
            acc = float(
                jnp.mean((jnp.argmax(logits, axis=-1) == y_test).astype(jnp.float32))
            )
            curve.append(acc)
            rows.append(
                {
                    "series": f"matrix_n={n}",
                    "epoch": epoch,
                    "test_accuracy": acc,
                    "params": count_params(proj_params) + count_params(readout),
                }
            )
        rows.append(
            {
                "series": f"matrix_n={n}_epoch10",
                "epoch": 10,
                "test_accuracy": curve[9],
                "params": count_params(proj_params) + count_params(readout),
            }
        )

    for epoch, acc in enumerate(scalar_curve, start=1):
        rows.append(
            {
                "series": "scalar",
                "epoch": epoch,
                "test_accuracy": acc,
                "params": count_params(scalar_params),
            }
        )

    save_rows_csv("examples/benchmarking_v2/outs/results_b4_mnist.csv", rows)
    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b4_accuracy.png",
        "B4 MNIST test accuracy",
        "epoch",
        "test accuracy",
        {
            series: (
                [float(r["epoch"]) for r in rows if r["series"] == series],
                [float(r["test_accuracy"]) for r in rows if r["series"] == series],
            )
            for series in ["scalar", "matrix_n=2", "matrix_n=4", "matrix_n=8"]
        },
    )
    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b4_epoch10_vs_n.png",
        "B4 sample efficiency (epoch 10)",
        "n",
        "test accuracy at epoch 10",
        {
            "matrix": (
                [2, 4, 8],
                [
                    float(
                        next(
                            r["test_accuracy"]
                            for r in rows
                            if r["series"] == f"matrix_n={n}_epoch10"
                        )
                    )
                    for n in [2, 4, 8]
                ],
            )
        },
    )
    print("B4 complete: outs/results_b4_mnist.csv and plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B4 MNIST benchmark")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b4(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
