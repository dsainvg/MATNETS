import jax
import jax.numpy as jnp

import matnets as mtn
from matnets import nn


def mlp_forward(params, x):
    x = mtn.dense(params["l1"], x, activation=jax.nn.relu)
    return mtn.dense(params["l2"], x)


def resblock(params, x):
    h = mtn.dense(params["w1"], x, activation=jax.nn.relu)
    h = mtn.dense(params["w2"], h)
    return jax.nn.relu(h + x)


def frobenius_attention_forward(params, x):
    project = jax.vmap(lambda token, p: mtn.dense(p, token), in_axes=(0, None))
    q = project(x, params["q"])
    k = project(x, params["k"])
    v = project(x, params["v"])
    return mtn.lax.matrix_attention(None, q, k, v)


def main():
    keys = jax.random.split(jax.random.key(7), 18)
    n = 3

    mlp_params = {
        "l1": mtn.init(keys[0], p=2, q=4, n=n),
        "l2": mtn.init(keys[1], p=4, q=1, n=n),
    }
    x = jax.random.normal(keys[2], (2, n, n))

    # Major parallel path: jit compiles both dense einsums into one staged function.
    mlp_y = jax.jit(mlp_forward)(mlp_params, x)
    print("mlp:", x.shape, "->", mlp_y.shape)

    batch = jax.random.normal(keys[3], (8, 2, n, n))
    # Major parallel path: vmap adds the leading batch axis around the same MLP.
    batched_mlp = jax.jit(jax.vmap(mlp_forward, in_axes=(None, 0)))
    batch_y = batched_mlp(mlp_params, batch)
    print("batched mlp:", batch.shape, "->", batch_y.shape)

    grad_params = jax.grad(lambda p, item: mlp_forward(p, item).sum())(mlp_params, x)
    print("mlp grad l1.W:", grad_params["l1"].W.shape)

    input_neurons = 2
    hidden_neurons = 4
    steps = 5
    seq = jax.random.normal(keys[4], (steps, input_neurons, n, n))
    h0 = jnp.zeros((hidden_neurons, n, n))

    rnn_params = mtn.init(
        keys[5],
        p=hidden_neurons + input_neurons,
        q=hidden_neurons,
        n=n,
    )
    # scan keeps time dependency explicit; each step's dense contraction is parallel.
    _, rnn_seq = jax.lax.scan(lambda h, x_t: nn.rnn_step(rnn_params, h, x_t), h0, seq)
    print("rnn seq:", seq.shape, "->", rnn_seq.shape)

    lstm_params = {
        "i": mtn.init(keys[6], p=hidden_neurons + input_neurons, q=hidden_neurons, n=n),
        "f": mtn.init(keys[7], p=hidden_neurons + input_neurons, q=hidden_neurons, n=n),
        "g": mtn.init(keys[8], p=hidden_neurons + input_neurons, q=hidden_neurons, n=n),
        "o": mtn.init(keys[9], p=hidden_neurons + input_neurons, q=hidden_neurons, n=n),
    }
    c0 = jnp.zeros_like(h0)
    _, lstm_seq = jax.lax.scan(
        lambda carry, x_t: nn.lstm_step(lstm_params, carry, x_t),
        (h0, c0),
        seq,
    )
    print("lstm seq:", seq.shape, "->", lstm_seq.shape)

    tokens = 6
    token_neurons = 3
    token_seq = jax.random.normal(keys[10], (tokens, token_neurons, n, n))
    attention_params = {
        "q": mtn.init(keys[11], p=token_neurons, q=token_neurons, n=n),
        "k": mtn.init(keys[12], p=token_neurons, q=token_neurons, n=n),
        "v": mtn.init(keys[13], p=token_neurons, q=token_neurons, n=n),
    }
    # Major parallel path: vmap projects tokens; nested vmap scores pairs.
    attention_y = jax.jit(frobenius_attention_forward)(attention_params, token_seq)
    print("attention:", token_seq.shape, "->", attention_y.shape)

    res_params = {
        "w1": mtn.init(keys[14], p=token_neurons, q=token_neurons, n=n),
        "w2": mtn.init(keys[15], p=token_neurons, q=token_neurons, n=n),
    }
    res_y = jax.jit(resblock)(res_params, token_seq[0])
    print("resblock:", token_seq[0].shape, "->", res_y.shape)


if __name__ == "__main__":
    main()
