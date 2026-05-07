"""B6 - Sequential modelling copy task benchmark."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import matnets as mtn
from examples.benchmarking_v2.common import (
    count_params,
    cross_entropy_from_logits,
    init_scalar_mlp,
    save_rows_csv,
    scalar_mlp_forward,
)


def _make_copy_batch(
    key: jax.Array,
    batch_size: int,
    seq_len: int = 20,
    delay: int = 10,
    vocab_size: int = 8,
) -> tuple[jax.Array, jax.Array]:
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    blanks = jnp.full((batch_size, delay), vocab_size)
    inp = jnp.concatenate([tokens, blanks], axis=1)
    target = jnp.concatenate(
        [jnp.full((batch_size, delay), vocab_size), tokens], axis=1
    )
    return inp, target


def run_b6(
    *,
    epochs: int = 1000,
    batch_size: int = 128,
    seq_len: int = 20,
    delay: int = 10,
    vocab_size: int = 8,
    seed: int = 0,
) -> None:
    vocab_total = vocab_size + 1
    scalar_hidden = 256

    scalar_in = init_scalar_mlp(
        jax.random.key(seed), [vocab_total + scalar_hidden, scalar_hidden, scalar_hidden]
    )
    scalar_out = init_scalar_mlp(jax.random.key(seed + 1), [scalar_hidden, vocab_total])

    def scalar_step(h, x_t):
        inp = jnp.concatenate([h, x_t], axis=-1)
        new_h = jnp.tanh(scalar_mlp_forward(scalar_in, inp))
        return new_h, scalar_mlp_forward(scalar_out, new_h)

    n = 4
    matrix_hidden = 32
    matrix_params = mtn.init(
        jax.random.key(seed + 2), p=matrix_hidden + 1, q=matrix_hidden, n=n
    )
    matrix_out = init_scalar_mlp(
        jax.random.key(seed + 3), [matrix_hidden * n * n, vocab_total]
    )

    @jax.jit
    def scalar_loss(p_in, p_out, x_tokens, y_tokens):
        x_oh = jax.nn.one_hot(x_tokens, vocab_total).astype(jnp.float32)

        def run_one(seq):
            h0 = jnp.zeros((scalar_hidden,))
            _, logits = jax.lax.scan(lambda h, x: scalar_step(h, x), h0, seq)
            return logits

        logits = jax.vmap(run_one)(x_oh)
        return cross_entropy_from_logits(
            logits.reshape(-1, vocab_total), y_tokens.reshape(-1), vocab_total
        )

    @jax.jit
    def matrix_loss(p_rec, p_out, x_tokens, y_tokens):
        x_oh = jax.nn.one_hot(x_tokens, vocab_total).astype(jnp.float32)

        def run_one(seq):
            h0 = jnp.zeros((matrix_hidden, n, n))

            def step(h, x):
                scalar_token = jnp.mean(x)
                x_as_matrix = jnp.full((1, n, n), scalar_token)
                combined = jnp.concatenate([h, x_as_matrix], axis=0)
                new_h = mtn.dense(p_rec, combined, activation=jnp.tanh)
                flat = new_h.reshape(-1)
                logits = scalar_mlp_forward(p_out, flat)
                return new_h, logits

            _, logits = jax.lax.scan(step, h0, seq)
            return logits

        logits = jax.vmap(run_one)(x_oh)
        return cross_entropy_from_logits(
            logits.reshape(-1, vocab_total), y_tokens.reshape(-1), vocab_total
        )

    grad_scalar = jax.jit(jax.grad(scalar_loss, argnums=(0, 1)))
    grad_matrix = jax.jit(jax.grad(matrix_loss, argnums=(0, 1)))
    lr = 1e-3
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        xb, yb = _make_copy_batch(
            jax.random.key(seed + epoch), batch_size, seq_len, delay, vocab_size
        )
        gs_in, gs_out = grad_scalar(scalar_in, scalar_out, xb, yb)
        scalar_in = jax.tree_util.tree_map(lambda p, g: p - lr * g, scalar_in, gs_in)
        scalar_out = jax.tree_util.tree_map(lambda p, g: p - lr * g, scalar_out, gs_out)
        gm_rec, gm_out = grad_matrix(matrix_params, matrix_out, xb, yb)
        matrix_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, matrix_params, gm_rec
        )
        matrix_out = jax.tree_util.tree_map(lambda p, g: p - lr * g, matrix_out, gm_out)

        if epoch % 25 == 0 or epoch == 1:
            l_s = float(scalar_loss(scalar_in, scalar_out, xb, yb))
            l_m = float(matrix_loss(matrix_params, matrix_out, xb, yb))
            rows.append(
                {
                    "epoch": epoch,
                    "model": "scalar",
                    "loss": l_s,
                    "params": count_params(scalar_in) + count_params(scalar_out),
                }
            )
            rows.append(
                {
                    "epoch": epoch,
                    "model": "matrix_n=4",
                    "loss": l_m,
                    "params": count_params(matrix_params) + count_params(matrix_out),
                }
            )

    save_rows_csv("examples/benchmarking_v2/outs/results_b6_copy_task.csv", rows)
    print("B6 complete: outs/results_b6_copy_task.csv saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B6 copy task benchmark")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b6(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        delay=args.delay,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
