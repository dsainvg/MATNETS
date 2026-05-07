import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matnets as mtn
from datasets import load_dataset
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import os

print("Starting RNN Benchmark Data Loading...")

def get_imdb_data(batch_size, max_seq_length=128, vocab_size=1000):
    # For a fast iteration, load IMDB and tokenize very simply
    dataset = load_dataset('imdb')
    train_data = dataset['train'].select(range(100))
    test_data = dataset['test'].select(range(100))

    def simple_tokenize(text):
        # Extremely rudimentary hashing tokenizer
        tokens = [hash(word) % vocab_size for word in text.split()[:max_seq_length]]
        if len(tokens) < max_seq_length:
            tokens.extend([0] * (max_seq_length - len(tokens))) # pad with 0
        return tokens

    X_train = np.array([simple_tokenize(text) for text in train_data['text']], dtype=np.int32)
    y_train = np.array(train_data['label'])

    X_test = np.array([simple_tokenize(text) for text in test_data['text']], dtype=np.int32)
    y_test = np.array(test_data['label'])

    return (X_train, y_train), (X_test, y_test), vocab_size




class ScalarLSTM(nn.Module):
    vocab_size: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        embeds = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(x)

        # define weights explicitly
        w_i = self.param('w_i', nn.initializers.lecun_normal(), (self.hidden_dim * 2, self.hidden_dim))
        b_i = self.param('b_i', nn.initializers.zeros, (self.hidden_dim,))
        w_f = self.param('w_f', nn.initializers.lecun_normal(), (self.hidden_dim * 2, self.hidden_dim))
        b_f = self.param('b_f', nn.initializers.zeros, (self.hidden_dim,))
        w_g = self.param('w_g', nn.initializers.lecun_normal(), (self.hidden_dim * 2, self.hidden_dim))
        b_g = self.param('b_g', nn.initializers.zeros, (self.hidden_dim,))
        w_o = self.param('w_o', nn.initializers.lecun_normal(), (self.hidden_dim * 2, self.hidden_dim))
        b_o = self.param('b_o', nn.initializers.zeros, (self.hidden_dim,))

        def scan_fn(carry, x_t):
            h, c = carry
            combined = jnp.concatenate([x_t, h], axis=-1)
            i = nn.sigmoid(jnp.dot(combined, w_i) + b_i)
            f = nn.sigmoid(jnp.dot(combined, w_f) + b_f)
            g = jnp.tanh(jnp.dot(combined, w_g) + b_g)
            o = nn.sigmoid(jnp.dot(combined, w_o) + b_o)
            next_c = f * c + i * g
            next_h = o * jnp.tanh(next_c)
            return (next_h, next_c), next_h

        init_h = jnp.zeros((x.shape[0], self.hidden_dim))
        init_c = jnp.zeros((x.shape[0], self.hidden_dim))

        embeds = jnp.transpose(embeds, (1, 0, 2))
        carry, outs = jax.lax.scan(scan_fn, (init_h, init_c), embeds)
        final_h, final_c = carry

        return nn.Dense(features=1)(final_h)

class MatNetLSTM(nn.Module):
    vocab_size: int
    hidden_neurons: int
    n: int

    @nn.compact
    def __call__(self, x):
        # x is (B, T)
        # Embed to (B, T, hidden_neurons, n, n)

        # We can simulate a matrix embedding by using standard embedding and reshaping,
        # or doing a dense projection.
        # For simplicity, use scalar embedding then reshape to matrix
        embed_dim = self.hidden_neurons * self.n * self.n
        embeds = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim)(x)
        embeds = embeds.reshape((x.shape[0], x.shape[1], self.hidden_neurons, self.n, self.n))

        lstm_params = {
            "i": mtn.MatrixParams(self.param("i_W", nn.initializers.lecun_normal(), (self.hidden_neurons, self.hidden_neurons*2, self.n, self.n)), self.param("i_B", nn.initializers.zeros, (self.hidden_neurons, self.n, self.n))),
            "f": mtn.MatrixParams(self.param("f_W", nn.initializers.lecun_normal(), (self.hidden_neurons, self.hidden_neurons*2, self.n, self.n)), self.param("f_B", nn.initializers.zeros, (self.hidden_neurons, self.n, self.n))),
            "g": mtn.MatrixParams(self.param("g_W", nn.initializers.lecun_normal(), (self.hidden_neurons, self.hidden_neurons*2, self.n, self.n)), self.param("g_B", nn.initializers.zeros, (self.hidden_neurons, self.n, self.n))),
            "o": mtn.MatrixParams(self.param("o_W", nn.initializers.lecun_normal(), (self.hidden_neurons, self.hidden_neurons*2, self.n, self.n)), self.param("o_B", nn.initializers.zeros, (self.hidden_neurons, self.n, self.n)))
        }

        def scan_fn(carry, x_t):
            return mtn.nn.lstm_step(lstm_params, carry, x_t)

        init_h = jnp.zeros((x.shape[0], self.hidden_neurons, self.n, self.n))
        init_c = jnp.zeros((x.shape[0], self.hidden_neurons, self.n, self.n))

        # Transpose to (T, B, hidden_neurons, n, n)
        embeds = jnp.transpose(embeds, (1, 0, 2, 3, 4))

        # jax.vmap the scan_fn over batch
        def batch_scan_fn(carry, x_t):
            # vmap over batch axis
            return jax.vmap(mtn.nn.lstm_step, in_axes=(None, 0, 0))(lstm_params, carry, x_t)

        carry, outs = jax.lax.scan(batch_scan_fn, (init_h, init_c), embeds)
        final_h, final_c = carry

        # Mean over hidden_neurons, n, n
        logits = final_h.mean(axis=(1, 2, 3))
        # Keep shape (B, 1)
        return jnp.expand_dims(logits, axis=-1)


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def benchmark_rnn():
    batch_size = 32
    epochs = 1

    (X_train, y_train), (X_test, y_test), vocab_size = get_imdb_data(batch_size)
    num_batches = len(X_train) // batch_size

    results = []

    for n in [8, 16]:
        hidden_neurons = 2

        key = jax.random.PRNGKey(0)
        dummy_seq = jax.random.randint(key, (1, 128), 0, vocab_size)

        mat_model = MatNetLSTM(vocab_size=vocab_size, hidden_neurons=hidden_neurons, n=n)
        mat_params = mat_model.init(key, dummy_seq)
        mat_param_count = count_params(mat_params)

        print(f"MATNETS (n={n}) Param Count: {mat_param_count}")

        hidden_dim = 16
        while True:
            scalar_model = ScalarLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim)
            scalar_params = scalar_model.init(key, dummy_seq)
            scalar_param_count = count_params(scalar_params)
            if scalar_param_count >= mat_param_count:
                break
            hidden_dim += 2

        print(f"Scalar Baseline (hidden_dim={hidden_dim}) Param Count: {scalar_param_count}")

        for model_type, model, params, is_mat in [
            ("MATNETS_LSTM", mat_model, mat_params, True),
            ("ScalarBaseline_LSTM", scalar_model, scalar_params, False)
        ]:
            tx = optax.adam(1e-3)
            state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

            @jax.jit
            def train_step(state, x_batch, y_batch):
                def loss_fn(p):
                    logits = state.apply_fn(p, x_batch)
                    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits.squeeze(), labels=y_batch.astype(jnp.float32)))
                    preds = (jax.nn.sigmoid(logits.squeeze()) > 0.5).astype(jnp.int32)
                    acc = jnp.mean(preds == y_batch)
                    return loss, acc
                (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                state = state.apply_gradients(grads=grads)
                return state, loss, acc

            dummy_x = jax.random.randint(key, (batch_size, 128), 0, vocab_size)
            dummy_y = jax.random.randint(key, (batch_size,), 0, 2)

            cost = train_step.lower(state, dummy_x, dummy_y).cost_analysis()
            flops = cost[0].get('flops', 0) if isinstance(cost, list) and len(cost) > 0 else cost.get('flops', 0) if isinstance(cost, dict) else 0

            # Warmup
            state, _, _ = train_step(state, dummy_x, dummy_y)

            total_time = 0
            for epoch in range(epochs):
                perm = np.random.permutation(len(X_train))
                X_train_shuf = X_train[perm]
                y_train_shuf = y_train[perm]

                start_time = time.time()
                epoch_loss = 0
                epoch_acc = 0

                for i in range(num_batches):
                    batch_x = X_train_shuf[i*batch_size:(i+1)*batch_size]
                    batch_y = y_train_shuf[i*batch_size:(i+1)*batch_size]

                    state, loss, acc = train_step(state, batch_x, batch_y)
                    epoch_loss += loss
                    epoch_acc += acc

                epoch_loss = epoch_loss.item() / num_batches
                epoch_acc = epoch_acc.item() / num_batches

                epoch_time = time.time() - start_time
                total_time += epoch_time

            results.append({
                "Task": "IMDB LSTM",
                "Model": model_type,
                "Size_n": n if is_mat else "-",
                "Params": count_params(state.params),
                "FLOPs": flops,
                "Time_per_Epoch": total_time / epochs,
                "Final_Loss": epoch_loss,
                "Final_Accuracy": epoch_acc
            })
            print(f"{model_type} n={n if is_mat else '-'} -> Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Time/epoch: {total_time/epochs:.4f}s")

    df = pd.DataFrame(results)
    if os.path.exists("summary_results.csv"):
        df.to_csv("summary_results.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("summary_results.csv", index=False)
    print("RNN Benchmarks saved to summary_results.csv")

if __name__ == "__main__":
    benchmark_rnn()
