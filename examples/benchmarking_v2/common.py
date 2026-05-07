"""Shared helpers for Benchmark Suite V2."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

import matnets as mtn
from matnets import MatrixParams

ArrayTree = Any


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    batch_size: int = 256
    epochs: int = 500
    learning_rate: float = 1e-3


def count_params(tree: ArrayTree) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(tree)))


def save_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        raise ValueError("rows must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_lines(
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: dict[str, tuple[list[float], list[float]]],
    *,
    xscale: str | None = None,
    yscale: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _block(x: Any) -> None:
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
        return
    leaves = jax.tree_util.tree_leaves(x)
    if leaves and hasattr(leaves[0], "block_until_ready"):
        leaves[0].block_until_ready()


def time_ms_median(fn, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        _block(fn())
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(median(times))


def peak_memory_mb() -> float | None:
    device = jax.devices()[0]
    stats = getattr(device, "memory_stats", None)
    if stats is None:
        return None
    raw = stats()
    if raw is None:
        return None
    for key in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit"):
        if key in raw and raw[key] is not None:
            return float(raw[key]) / (1024.0 * 1024.0)
    return None


def init_scalar_mlp(
    key: Array, dims: list[int], scale: float = 0.02
) -> list[dict[str, Array]]:
    keys = jax.random.split(key, len(dims) - 1)
    out: list[dict[str, Array]] = []
    for k, (d_in, d_out) in zip(
        keys,
        zip(dims[:-1], dims[1:], strict=True),
        strict=True,
    ):
        kw, kb = jax.random.split(k)
        out.append(
            {
                "W": scale * jax.random.normal(kw, (d_out, d_in)),
                "b": jnp.zeros((d_out,)),
            }
        )
    return out


def scalar_mlp_forward(
    params: list[dict[str, Array]], x: Array, *, relu_last: bool = False
) -> Array:
    h = x
    for i, layer in enumerate(params):
        h = jnp.dot(layer["W"], h) + layer["b"]
        if i < len(params) - 1 or relu_last:
            h = jax.nn.relu(h)
    return h


def init_matrix_mlp(key: Array, neurons: list[int], n: int) -> list[MatrixParams]:
    keys = jax.random.split(key, len(neurons) - 1)
    return [
        mtn.init(k, p=d_in, q=d_out, n=n)
        for k, (d_in, d_out) in zip(
            keys,
            zip(neurons[:-1], neurons[1:], strict=True),
            strict=True,
        )
    ]


def matrix_mlp_forward(
    params: list[MatrixParams], x: Array, *, relu_last: bool = False
) -> Array:
    h = x
    for i, layer in enumerate(params):
        use_relu = i < len(params) - 1 or relu_last
        h = mtn.dense(layer, h, activation=jax.nn.relu if use_relu else (lambda t: t))
    return h


def mse(pred: Array, target: Array) -> Array:
    return jnp.mean((pred - target) ** 2)


def one_hot(y: Array, num_classes: int) -> Array:
    return jax.nn.one_hot(y, num_classes=num_classes)


def cross_entropy_from_logits(logits: Array, y: Array, num_classes: int) -> Array:
    y_oh = one_hot(y, num_classes=num_classes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(y_oh * log_probs, axis=-1))
