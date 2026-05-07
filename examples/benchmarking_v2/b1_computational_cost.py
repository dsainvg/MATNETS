"""B1 - Computational Cost benchmark."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from examples.benchmarking_v2.common import (
    count_params,
    init_matrix_mlp,
    init_scalar_mlp,
    matrix_mlp_forward,
    maybe_plot_lines,
    peak_memory_mb,
    save_rows_csv,
    scalar_mlp_forward,
    time_ms_median,
)


def run_b1(
    *,
    repeats: int = 1000,
    batch_size: int = 256,
    seed: int = 0,
) -> None:
    ns = [2, 4, 8, 16, 32]
    scalar_dims = [512, 512, 512, 512, 128]
    scalar_key = jax.random.key(seed)
    scalar_params = init_scalar_mlp(scalar_key, scalar_dims)
    scalar_param_count = count_params(scalar_params)
    xb_scalar = jax.random.normal(jax.random.key(seed + 1), (batch_size, 512))
    yb_scalar = jax.random.normal(jax.random.key(seed + 2), (batch_size, 128))

    def scalar_loss(p, xb, yb):
        pred = jax.vmap(lambda x: scalar_mlp_forward(p, x))(xb)
        return jnp.mean((pred - yb) ** 2)

    scalar_fwd_jit = jax.jit(
        lambda p, xb: jax.vmap(lambda x: scalar_mlp_forward(p, x))(xb)
    )
    scalar_step_jit = jax.jit(jax.value_and_grad(scalar_loss))

    def scalar_fwd_call() -> jax.Array:
        return scalar_fwd_jit(scalar_params, xb_scalar)

    def scalar_step_call() -> tuple[jax.Array, object]:
        return scalar_step_jit(scalar_params, xb_scalar, yb_scalar)

    scalar_fwd_ms = time_ms_median(scalar_fwd_call, repeats)
    scalar_step_ms = time_ms_median(scalar_step_call, repeats)
    scalar_mem_mb = peak_memory_mb()

    rows: list[dict[str, float | int | str | None]] = []
    rows.append(
        {
            "model": "scalar_baseline",
            "n": 0,
            "params": scalar_param_count,
            "forward_ms": scalar_fwd_ms,
            "step_ms": scalar_step_ms,
            "peak_memory_mb": scalar_mem_mb,
        }
    )

    for n in ns:
        key = jax.random.key(seed + 100 + n)
        matrix_params = init_matrix_mlp(key, [512, 512, 512, 512, 128], n)
        xb = jax.random.normal(jax.random.key(seed + 200 + n), (batch_size, 512, n, n))
        yb = jax.random.normal(jax.random.key(seed + 300 + n), (batch_size, 128, n, n))

        def matrix_loss(p, x_batch, y_batch):
            pred = jax.vmap(lambda x: matrix_mlp_forward(p, x))(x_batch)
            return jnp.mean((pred - y_batch) ** 2)

        fwd_jit = jax.jit(
            lambda p, x_batch: jax.vmap(lambda x: matrix_mlp_forward(p, x))(x_batch)
        )
        step_jit = jax.jit(jax.value_and_grad(matrix_loss))

        def matrix_fwd_call(p=matrix_params, x_batch=xb, f_jit=fwd_jit) -> jax.Array:
            return f_jit(p, x_batch)

        def matrix_step_call(
            p=matrix_params,
            x_batch=xb,
            y_batch=yb,
            s_jit=step_jit,
        ) -> tuple[jax.Array, object]:
            return s_jit(p, x_batch, y_batch)

        fwd_ms = time_ms_median(matrix_fwd_call, repeats)
        step_ms = time_ms_median(matrix_step_call, repeats)
        mem_mb = peak_memory_mb()

        rows.append(
            {
                "model": "matrix",
                "n": n,
                "params": count_params(matrix_params),
                "forward_ms": fwd_ms,
                "step_ms": step_ms,
                "peak_memory_mb": mem_mb,
            }
        )

    save_rows_csv("examples/benchmarking_v2/outs/results_b1_cost.csv", rows)

    xs = ns
    forward_ys = [float(r["forward_ms"]) for r in rows if r["model"] == "matrix"]
    step_ys = [float(r["step_ms"]) for r in rows if r["model"] == "matrix"]
    mem_ys = [float(r["peak_memory_mb"] or 0.0) for r in rows if r["model"] == "matrix"]
    scalar_fwd = [scalar_fwd_ms for _ in ns]
    scalar_step = [scalar_step_ms for _ in ns]
    scalar_mem = [(scalar_mem_mb or 0.0) for _ in ns]

    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b1_forward_ms.png",
        "B1 Forward Time vs n",
        "matrix dimension n",
        "ms/forward",
        {
            "matrix": (xs, forward_ys),
            "scalar baseline": (xs, scalar_fwd),
        },
    )
    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b1_step_ms.png",
        "B1 Forward+Backward Time vs n",
        "matrix dimension n",
        "ms/step",
        {
            "matrix": (xs, step_ys),
            "scalar baseline": (xs, scalar_step),
        },
    )
    maybe_plot_lines(
        "examples/benchmarking_v2/outs/plot_b1_memory_mb.png",
        "B1 Peak Memory vs n",
        "matrix dimension n",
        "peak memory (MB)",
        {
            "matrix": (xs, mem_ys),
            "scalar baseline": (xs, scalar_mem),
        },
    )

    print("B1 complete: outs/results_b1_cost.csv and plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B1 computational cost benchmark")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b1(repeats=args.repeats, batch_size=args.batch_size, seed=args.seed)
