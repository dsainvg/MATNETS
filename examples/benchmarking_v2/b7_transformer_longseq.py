"""B7 - Transformer long sequence benchmark."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from examples.benchmarking_v2.common import (
    count_params,
    cross_entropy_from_logits,
    init_scalar_mlp,
    save_rows_csv,
    scalar_mlp_forward,
)


def _make_synthetic_lm_batch(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    vocab_size: int = 256,
) -> tuple[jax.Array, jax.Array]:
    x = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    y = jnp.roll(x, shift=-1, axis=1)
    return x, y


def run_b7(
    *,
    epochs: int = 200,
    batch_size: int = 16,
    seq_len: int = 4096,
    seed: int = 0,
) -> None:
    vocab = 256

    # Scalar transformer proxy: token embedding + MLP readout on each token.
    emb = init_scalar_mlp(jax.random.key(seed), [vocab, 512])
    model = init_scalar_mlp(jax.random.key(seed + 1), [512, 512, 512, vocab])

    # Matrix transformer proxy: reshape token feature to matrix and use matrix MLP.
    n = 8
    matrix_proj = init_scalar_mlp(jax.random.key(seed + 2), [vocab, 64 * n * n])
    matrix_model = init_scalar_mlp(
        jax.random.key(seed + 3), [64 * n * n, 64 * n * n, vocab]
    )

    @jax.jit
    def scalar_loss(emb_p, mdl_p, x_tok, y_tok):
        x_oh = jax.nn.one_hot(x_tok, vocab).astype(jnp.float32)
        h = jax.vmap(jax.vmap(lambda v: scalar_mlp_forward(emb_p, v)))(x_oh)
        logits = jax.vmap(jax.vmap(lambda v: scalar_mlp_forward(mdl_p, v)))(h)
        return cross_entropy_from_logits(
            logits.reshape(-1, vocab), y_tok.reshape(-1), vocab
        )

    @jax.jit
    def matrix_loss(proj_p, mdl_p, x_tok, y_tok):
        x_oh = jax.nn.one_hot(x_tok, vocab).astype(jnp.float32)
        h = jax.vmap(jax.vmap(lambda v: scalar_mlp_forward(proj_p, v)))(x_oh)
        logits = jax.vmap(jax.vmap(lambda v: scalar_mlp_forward(mdl_p, v)))(h)
        return cross_entropy_from_logits(
            logits.reshape(-1, vocab), y_tok.reshape(-1), vocab
        )

    gs = jax.jit(jax.grad(scalar_loss, argnums=(0, 1)))
    gm = jax.jit(jax.grad(matrix_loss, argnums=(0, 1)))
    lr = 1e-3

    rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        xb, yb = _make_synthetic_lm_batch(
            jax.random.key(seed + epoch), batch_size, seq_len, vocab
        )
        g_emb, g_model = gs(emb, model, xb, yb)
        emb = jax.tree_util.tree_map(lambda p, g: p - lr * g, emb, g_emb)
        model = jax.tree_util.tree_map(lambda p, g: p - lr * g, model, g_model)
        g_proj, g_m = gm(matrix_proj, matrix_model, xb, yb)
        matrix_proj = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, matrix_proj, g_proj
        )
        matrix_model = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, matrix_model, g_m
        )
        if epoch % 10 == 0 or epoch == 1:
            l_s = float(scalar_loss(emb, model, xb, yb))
            l_m = float(matrix_loss(matrix_proj, matrix_model, xb, yb))
            rows.append(
                {
                    "epoch": epoch,
                    "model": "scalar_transformer_proxy",
                    "loss": l_s,
                    "params": count_params(emb) + count_params(model),
                }
            )
            rows.append(
                {
                    "epoch": epoch,
                    "model": "matrix_transformer_proxy_n=8",
                    "loss": l_m,
                    "params": count_params(matrix_proj) + count_params(matrix_model),
                }
            )

    save_rows_csv("examples/benchmarking_v2/outs/results_b7_transformer_longseq.csv", rows)
    print("B7 complete: outs/results_b7_transformer_longseq.csv saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B7 transformer long sequence benchmark"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_b7(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
    )
