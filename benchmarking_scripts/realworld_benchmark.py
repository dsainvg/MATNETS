
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matnets as mtn
import numpy as np
import time
import pandas as pd
import os
from tqdm import tqdm
from datasets import load_dataset
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.preprocessing import StandardScaler

print("Starting Real-World Benchmark Data Loading...")

def get_realworld_data(task):
    if task == "LINEAR_REGRESSION":
        data = fetch_california_housing()
        X = StandardScaler().fit_transform(data.data).astype(np.float32)
        y = data.target.astype(np.float32)
        return X, y, X.shape[1], 1

    elif task == "TABULAR":
        data = load_breast_cancer()
        X = StandardScaler().fit_transform(data.data).astype(np.float32)
        y = data.target.astype(np.int32)
        return X, y, X.shape[1], 2

    elif task == "CONV2D":
        # Keep small subset for fast testing but still real data
        dataset = load_dataset('cifar10')
        train_data = dataset['train']
        # Convert CIFAR to flat grayscale or just use 1 channel to map to our benchmark
        # To avoid massive memory issues, let's use MNIST here
        dataset = load_dataset('mnist')
        train_data = dataset['train']
        X = np.array(train_data['image']).astype(np.float32) / 255.0
        X = X.reshape(-1, 28, 28, 1)
        y = np.array(train_data['label'])
        return X, y, 1, 10

    elif task == "CONV1D":
        dataset = load_dataset('mnist')
        train_data = dataset['train']
        X = np.array(train_data['image']).astype(np.float32) / 255.0
        X = X.reshape(-1, 784, 1)
        y = np.array(train_data['label'])
        return X, y, 1, 10

    elif task == "SEQUENCE":
        dataset = load_dataset('imdb')
        train_data = dataset['train']
        vocab_size = 5000
        max_seq_length = 128

        def simple_tokenize(text):
            tokens = [hash(word) % vocab_size for word in text.split()[:max_seq_length]]
            if len(tokens) < max_seq_length:
                tokens.extend([0] * (max_seq_length - len(tokens)))
            return tokens

        X = np.array([simple_tokenize(text) for text in train_data['text']], dtype=np.int32)
        y = np.array(train_data['label'])
        return X, y, vocab_size, 2

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# --- ARCHITECTURES ---
class MatNetLinearRegression(nn.Module):
    n: int
    input_dim: int
    @nn.compact
    def __call__(self, x):
        # x: (B, input_dim) -> map to (B, input_dim, n, n)
        x_mapped = jnp.tile(x[..., None, None], (1, 1, self.n, self.n))
        w = self.param("W", nn.initializers.lecun_normal(), (1, self.input_dim, self.n, self.n))
        b = self.param("B", nn.initializers.zeros, (1, self.n, self.n))
        out = jax.vmap(lambda x_t: mtn.dense(mtn.MatrixParams(W=w, B=b), x_t))(x_mapped)
        return out.mean(axis=(1, 2, 3))

class ScalarLinearRegression(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        return nn.Dense(1)(x).squeeze()

class MatNetDense(nn.Module):
    n: int
    input_dim: int
    @nn.compact
    def __call__(self, x):
        x_mapped = jnp.tile(x[..., None, None], (1, 1, self.n, self.n))
        w1 = self.param("W1", nn.initializers.lecun_normal(), (4, self.input_dim, self.n, self.n))
        b1 = self.param("B1", nn.initializers.zeros, (4, self.n, self.n))
        w2 = self.param("W2", nn.initializers.lecun_normal(), (1, 4, self.n, self.n))
        b2 = self.param("B2", nn.initializers.zeros, (1, self.n, self.n))
        h = jax.nn.relu(jax.vmap(lambda x_t: mtn.dense(mtn.MatrixParams(W=w1, B=b1), x_t))(x_mapped))
        out = jax.vmap(lambda h_t: mtn.dense(mtn.MatrixParams(W=w2, B=b2), h_t))(h)
        return out.mean(axis=(1, 2, 3))

class ScalarDense(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.hidden_dim)(x)
        return nn.Dense(1)(x).squeeze()

class MatNetConv1D(nn.Module):
    n: int
    p: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        x_mapped = jnp.tile(x[..., None, None], (1, 1, 1, self.n, self.n))
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, self.p, 3, self.n, self.n)),
            B=self.param("c1_B", nn.initializers.zeros, (4, self.n, self.n))
        )
        h = jax.nn.relu(jax.vmap(lambda p_c, x_t: mtn.lax.matrix_conv1d(p_c, x_t, padding="SAME"), in_axes=(None, 0))(c1, x_mapped))
        outp = mtn.MatrixParams(W=self.param("out_W", nn.initializers.lecun_normal(), (self.num_classes, 4, self.n, self.n)), B=self.param("out_B", nn.initializers.zeros, (self.num_classes, self.n, self.n)))
        out = jax.vmap(mtn.dense, in_axes=(None, 0))(outp, h.mean(axis=1))
        return out.mean(axis=(2, 3))

class ScalarConv1D(nn.Module):
    hidden_dim: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(self.hidden_dim, kernel_size=(3,), padding="SAME")(x))
        x = nn.Conv(self.hidden_dim, kernel_size=(3,), padding="SAME")(x)
        out = nn.Dense(self.num_classes)(x.mean(axis=1))
        return out.squeeze() if self.num_classes == 1 else out

class MatNetConv2D(nn.Module):
    n: int
    p: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        x_mapped = jnp.tile(x[..., None, None], (1, 1, 1, 1, self.n, self.n))
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, self.p, 3, 3, self.n, self.n)),
            B=self.param("c1_B", nn.initializers.zeros, (4, self.n, self.n))
        )
        h = jax.nn.relu(jax.vmap(lambda p_c, x_t: mtn.lax.matrix_conv2d(p_c, x_t, padding="SAME"), in_axes=(None, 0))(c1, x_mapped))
        outp = mtn.MatrixParams(W=self.param("out_W", nn.initializers.lecun_normal(), (self.num_classes, 4, self.n, self.n)), B=self.param("out_B", nn.initializers.zeros, (self.num_classes, self.n, self.n)))
        out = jax.vmap(mtn.dense, in_axes=(None, 0))(outp, h.mean(axis=(1, 2)))
        return out.mean(axis=(2, 3))

class ScalarConv2D(nn.Module):
    hidden_dim: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(self.hidden_dim, kernel_size=(3,3), padding="SAME")(x))
        x = nn.Conv(self.hidden_dim, kernel_size=(3,3), padding="SAME")(x)
        out = nn.Dense(self.num_classes)(x.mean(axis=(1, 2)))
        return out.squeeze() if self.num_classes == 1 else out

class MatNetRNN(nn.Module):
    n: int
    num_classes: int
    vocab_size: int
    @nn.compact
    def __call__(self, x):
        embed_dim = 2 * self.n * self.n
        embeds = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim)(x)
        embeds = embeds.reshape((x.shape[0], x.shape[1], 2, self.n, self.n))
        embeds = jnp.transpose(embeds, (1, 0, 2, 3, 4))

        rnn_params = mtn.MatrixParams(
            W=self.param("W", nn.initializers.lecun_normal(), (2, 2, self.n, self.n)),
            B=self.param("B", nn.initializers.zeros, (2, self.n, self.n))
        )
        def scan_fn(carry, x_t):
            return jax.vmap(mtn.nn.rnn_step, in_axes=(None, 0, 0))(rnn_params, carry, x_t)
        init_h = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        carry, _ = jax.lax.scan(scan_fn, init_h, embeds)
        outp = mtn.MatrixParams(W=self.param("out_W", nn.initializers.lecun_normal(), (self.num_classes, 2, self.n, self.n)), B=self.param("out_B", nn.initializers.zeros, (self.num_classes, self.n, self.n)))
        out = jax.vmap(mtn.dense, in_axes=(None, 0))(outp, carry)
        return out.mean(axis=(2, 3))

class ScalarRNN(nn.Module):
    hidden_dim: int
    num_classes: int
    vocab_size: int
    @nn.compact
    def __call__(self, x):
        embeds = nn.Embed(num_embeddings=self.vocab_size, features=16)(x)
        embeds = jnp.transpose(embeds, (1, 0, 2))

        w_h = self.param('w_h', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_h = self.param('b_h', nn.initializers.zeros, (self.hidden_dim,))
        def scan_fn(carry, x_t):
            combined = jnp.concatenate([carry, x_t], axis=-1)
            next_h = jnp.tanh(jnp.dot(combined, w_h) + b_h)
            return next_h, next_h
        init_h = jnp.zeros((embeds.shape[1], self.hidden_dim))
        carry, _ = jax.lax.scan(scan_fn, init_h, embeds)
        out = nn.Dense(self.num_classes)(carry)
        return out.squeeze() if self.num_classes == 1 else out

class MatNetLSTM(nn.Module):
    n: int
    num_classes: int
    vocab_size: int
    @nn.compact
    def __call__(self, x):
        embed_dim = 2 * self.n * self.n
        embeds = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim)(x)
        embeds = embeds.reshape((x.shape[0], x.shape[1], 2, self.n, self.n))
        embeds = jnp.transpose(embeds, (1, 0, 2, 3, 4))

        lstm_params = {
            "i": mtn.MatrixParams(self.param("i_W", nn.initializers.lecun_normal(), (2, 4, self.n, self.n)), self.param("i_B", nn.initializers.zeros, (2, self.n, self.n))),
            "f": mtn.MatrixParams(self.param("f_W", nn.initializers.lecun_normal(), (2, 4, self.n, self.n)), self.param("f_B", nn.initializers.zeros, (2, self.n, self.n))),
            "g": mtn.MatrixParams(self.param("g_W", nn.initializers.lecun_normal(), (2, 4, self.n, self.n)), self.param("g_B", nn.initializers.zeros, (2, self.n, self.n))),
            "o": mtn.MatrixParams(self.param("o_W", nn.initializers.lecun_normal(), (2, 4, self.n, self.n)), self.param("o_B", nn.initializers.zeros, (2, self.n, self.n)))
        }
        def batch_scan_fn(carry, x_t):
            return jax.vmap(mtn.nn.lstm_step, in_axes=(None, 0, 0))(lstm_params, carry, x_t)
        init_h = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        init_c = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        carry, _ = jax.lax.scan(batch_scan_fn, (init_h, init_c), embeds)
        outp = mtn.MatrixParams(W=self.param("out_W", nn.initializers.lecun_normal(), (self.num_classes, 2, self.n, self.n)), B=self.param("out_B", nn.initializers.zeros, (self.num_classes, self.n, self.n)))
        out = jax.vmap(mtn.dense, in_axes=(None, 0))(outp, carry[0])
        return out.mean(axis=(2, 3))

class ScalarLSTM(nn.Module):
    hidden_dim: int
    num_classes: int
    vocab_size: int
    @nn.compact
    def __call__(self, x):
        embeds = nn.Embed(num_embeddings=self.vocab_size, features=16)(x)
        embeds = jnp.transpose(embeds, (1, 0, 2))
        w_i = self.param('w_i', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_i = self.param('b_i', nn.initializers.zeros, (self.hidden_dim,))
        w_f = self.param('w_f', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_f = self.param('b_f', nn.initializers.zeros, (self.hidden_dim,))
        w_g = self.param('w_g', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_g = self.param('b_g', nn.initializers.zeros, (self.hidden_dim,))
        w_o = self.param('w_o', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
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
        init_h = jnp.zeros((embeds.shape[1], self.hidden_dim))
        init_c = jnp.zeros((embeds.shape[1], self.hidden_dim))
        carry, _ = jax.lax.scan(scan_fn, (init_h, init_c), embeds)
        out = nn.Dense(self.num_classes)(carry[0])
        return out.squeeze() if self.num_classes == 1 else out

def get_loss_fn(task, state, y_batch, num_classes):
    def loss_fn(p, x_batch):
        logits = state.apply_fn(p, x_batch)
        if num_classes > 2:
            one_hot = jax.nn.one_hot(y_batch, num_classes)
            return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        elif task == "LINEAR_REGRESSION":
            return jnp.mean(optax.l2_loss(predictions=logits.squeeze(), targets=y_batch))
        else:
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits.squeeze(), labels=y_batch.astype(jnp.float32)))
    return loss_fn

def run_massive_benchmark():
    results = []
    tasks = {
        "LINEAR_REGRESSION": (MatNetLinearRegression, ScalarLinearRegression),
        "TABULAR": (MatNetDense, ScalarDense),
        "CONV1D": (MatNetConv1D, ScalarConv1D),
        "CONV2D": (MatNetConv2D, ScalarConv2D),
        "RNN": (MatNetRNN, ScalarRNN),
        "LSTM": (MatNetLSTM, ScalarLSTM)
    }

    n_values = [4, 8]
    batch_sizes = [32, 64]
    learning_rates = [1e-3, 5e-4]

    key = jax.random.PRNGKey(42)

    print("Pre-fetching all datasets to memory...")
    task_data = {}
    for t in tasks.keys():
        task_category = "SEQUENCE" if t in ["RNN", "LSTM"] else t
        if task_category not in task_data:
            print(f"Loading data for {task_category}...")
            task_data[task_category] = get_realworld_data(task_category)

    print("Starting Massive Real-World Grid Search...")

    for task_name, (MatNetClass, ScalarClass) in tasks.items():
        data_category = "SEQUENCE" if task_name in ["RNN", "LSTM"] else task_name
        X_full, y_full, input_dim, num_classes = task_data[data_category]

        for n in n_values:
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    epochs = 2
                    num_batches = len(X_full) // batch_size
                    if num_batches == 0: continue

                    try:
                        dummy_x = X_full[:batch_size]

                        if task_name in ["CONV1D", "CONV2D"]:
                            mat_model = MatNetClass(n=n, p=input_dim, num_classes=num_classes)
                        elif task_name in ["TABULAR", "LINEAR_REGRESSION"]:
                            mat_model = MatNetClass(n=n, input_dim=input_dim)
                        else:
                            mat_model = MatNetClass(n=n, num_classes=num_classes, vocab_size=input_dim)

                        mat_params = mat_model.init(key, dummy_x)
                        mat_param_count = count_params(mat_params)

                        hidden_dim = 4
                        while True:
                            if task_name in ["TABULAR", "LINEAR_REGRESSION"]:
                                scalar_model = ScalarClass(hidden_dim=hidden_dim)
                            else:
                                scalar_model = ScalarClass(hidden_dim=hidden_dim, num_classes=num_classes)

                            if task_name in ["RNN", "LSTM"]:
                                scalar_model = ScalarClass(hidden_dim=hidden_dim, num_classes=num_classes, vocab_size=input_dim)

                            scalar_params = scalar_model.init(key, dummy_x)
                            if count_params(scalar_params) >= mat_param_count:
                                break
                            hidden_dim += 2

                        for model_type, model, params, is_mat in [
                            ("MATNETS", mat_model, mat_params, True),
                            ("Scalar", scalar_model, scalar_params, False)
                        ]:
                            tx = optax.adam(lr)
                            state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

                            @jax.jit
                            def train_step(state, x_batch, y_batch):
                                loss_fn = get_loss_fn(task_name, state, y_batch, num_classes)
                                loss, grads = jax.value_and_grad(loss_fn)(state.params, x_batch)
                                return state.apply_gradients(grads=grads), loss

                            dummy_y = y_full[:batch_size]

                            cost = train_step.lower(state, dummy_x, dummy_y).cost_analysis()
                            flops = cost[0].get('flops', 0) if isinstance(cost, list) and len(cost) > 0 else cost.get('flops', 0) if isinstance(cost, dict) else 0
                            state, _ = train_step(state, dummy_x, dummy_y)

                            start_time = time.time()
                            final_loss = 0
                            for ep in range(epochs):
                                perm = np.random.permutation(len(X_full))
                                X_shuf = X_full[perm]
                                y_shuf = y_full[perm]

                                for i in range(num_batches):
                                    bx = X_shuf[i*batch_size:(i+1)*batch_size]
                                    by = y_shuf[i*batch_size:(i+1)*batch_size]
                                    state, l = train_step(state, bx, by)
                                    final_loss = l

                            total_time = time.time() - start_time

                            results.append({
                                "Architecture": task_name,
                                "Model": model_type,
                                "Size_n": n if is_mat else "-",
                                "BatchSize": batch_size,
                                "LearningRate": lr,
                                "Params": count_params(state.params),
                                "FLOPs_per_step": flops,
                                "Total_Time": total_time,
                                "Time_per_Epoch": total_time / epochs,
                                "Final_Loss": float(final_loss)
                            })
                            print(f"{task_name} | {model_type} n={n if is_mat else '-'} b={batch_size} lr={lr} | Time: {total_time:.1f}s, Loss: {final_loss:.4f}")

                            df_temp = pd.DataFrame([results[-1]])
                            if os.path.exists("realworld_results.csv"):
                                df_temp.to_csv("realworld_results.csv", mode='a', header=False, index=False)
                            else:
                                df_temp.to_csv("realworld_results.csv", index=False)

                    except Exception as e:
                        print(f"Skipped {task_name} due to error: {e}")

if __name__ == "__main__":
    run_massive_benchmark()
