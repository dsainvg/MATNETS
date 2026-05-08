
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

def get_dummy_data(batch_size, task, seq_len=64, input_dim=10):
    if task == "TABULAR":
        X = np.random.randn(batch_size, input_dim).astype(np.float32)
        y = np.random.randint(0, 2, size=(batch_size,))
    elif task == "CONV1D":
        X = np.random.randn(batch_size, seq_len, 1).astype(np.float32)
        y = np.random.randint(0, 2, size=(batch_size,))
    elif task == "CONV2D":
        X = np.random.randn(batch_size, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, size=(batch_size,))
    elif task == "SEQUENCE":
        X = np.random.randint(0, 100, size=(batch_size, seq_len)).astype(np.int32)
        y = np.random.randint(0, 2, size=(batch_size,))
    elif task == "LINEAR_REGRESSION":
        X = np.random.randn(batch_size, input_dim).astype(np.float32)
        y = np.sum(X, axis=1) + np.random.randn(batch_size) * 0.1
    return X, y

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

class MatNetLinearRegression(nn.Module):
    n: int
    input_dim: int
    @nn.compact
    def __call__(self, x):
        w = self.param("W", nn.initializers.lecun_normal(), (1, self.input_dim, self.n, self.n))
        b = self.param("B", nn.initializers.zeros, (1, self.n, self.n))
        out = jax.vmap(lambda x_t: mtn.dense(mtn.MatrixParams(W=w, B=b), x_t))(x)
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
        w1 = self.param("W1", nn.initializers.lecun_normal(), (4, self.input_dim, self.n, self.n))
        b1 = self.param("B1", nn.initializers.zeros, (4, self.n, self.n))
        w2 = self.param("W2", nn.initializers.lecun_normal(), (1, 4, self.n, self.n))
        b2 = self.param("B2", nn.initializers.zeros, (1, self.n, self.n))
        h = jax.nn.relu(jax.vmap(lambda x_t: mtn.dense(mtn.MatrixParams(W=w1, B=b1), x_t))(x))
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
    @nn.compact
    def __call__(self, x):
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, self.p, 3, self.n, self.n)),
            B=self.param("c1_B", nn.initializers.zeros, (4, self.n, self.n))
        )
        h = jax.nn.relu(jax.vmap(lambda p_c, x_t: mtn.lax.matrix_conv1d(p_c, x_t, padding="SAME"), in_axes=(None, 0))(c1, x))
        return h.mean(axis=(1, 2, 3, 4))

class ScalarConv1D(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(self.hidden_dim, kernel_size=(3,), padding="SAME")(x))
        x = nn.Conv(self.hidden_dim, kernel_size=(3,), padding="SAME")(x)
        return nn.Dense(1)(x.mean(axis=(1, 2))).squeeze()

class MatNetConv2D(nn.Module):
    n: int
    p: int
    @nn.compact
    def __call__(self, x):
        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (4, self.p, 3, 3, self.n, self.n)),
            B=self.param("c1_B", nn.initializers.zeros, (4, self.n, self.n))
        )
        h = jax.nn.relu(jax.vmap(lambda p_c, x_t: mtn.lax.matrix_conv2d(p_c, x_t, padding="SAME"), in_axes=(None, 0))(c1, x))
        outp = mtn.MatrixParams(W=self.param("out_W", nn.initializers.lecun_normal(), (10, 4, self.n, self.n)), B=self.param("out_B", nn.initializers.zeros, (10, self.n, self.n)))
        out = jax.vmap(mtn.dense, in_axes=(None, 0))(outp, h.mean(axis=(1, 2)))
        return out.mean(axis=(2, 3))

class ScalarConv2D(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(self.hidden_dim, kernel_size=(3,3), padding="SAME")(x))
        x = nn.Conv(self.hidden_dim, kernel_size=(3,3), padding="SAME")(x)
        return nn.Dense(10)(x.mean(axis=(1, 2)))

class MatNetRNN(nn.Module):
    n: int
    @nn.compact
    def __call__(self, embeds):
        rnn_params = mtn.MatrixParams(
            W=self.param("W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)),
            B=self.param("B", nn.initializers.zeros, (2, self.n, self.n))
        )
        def scan_fn(carry, x_t):
            return jax.vmap(mtn.nn.rnn_step, in_axes=(None, 0, 0))(rnn_params, carry, x_t)
        init_h = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        carry, _ = jax.lax.scan(scan_fn, init_h, embeds)
        return carry.mean(axis=(1, 2, 3))

class ScalarRNN(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, embeds):
        w_h = self.param('w_h', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_h = self.param('b_h', nn.initializers.zeros, (self.hidden_dim,))
        def scan_fn(carry, x_t):
            combined = jnp.concatenate([carry, x_t], axis=-1)
            next_h = jnp.tanh(jnp.dot(combined, w_h) + b_h)
            return next_h, next_h
        init_h = jnp.zeros((embeds.shape[1], self.hidden_dim))
        carry, _ = jax.lax.scan(scan_fn, init_h, embeds)
        return nn.Dense(1)(carry).squeeze()

class MatNetGRU(nn.Module):
    n: int
    @nn.compact
    def __call__(self, embeds):
        gru_params = {
            "z": mtn.MatrixParams(self.param("z_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("z_B", nn.initializers.zeros, (2, self.n, self.n))),
            "r": mtn.MatrixParams(self.param("r_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("r_B", nn.initializers.zeros, (2, self.n, self.n))),
            "n": mtn.MatrixParams(self.param("n_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("n_B", nn.initializers.zeros, (2, self.n, self.n)))
        }
        def scan_fn(carry, x_t):
            return jax.vmap(mtn.nn.gru_step, in_axes=(None, 0, 0))(gru_params, carry, x_t)
        init_h = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        carry, _ = jax.lax.scan(scan_fn, init_h, embeds)
        return carry.mean(axis=(1, 2, 3))

class ScalarGRU(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, embeds):
        w_z = self.param('w_z', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_z = self.param('b_z', nn.initializers.zeros, (self.hidden_dim,))
        w_r = self.param('w_r', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_r = self.param('b_r', nn.initializers.zeros, (self.hidden_dim,))
        w_n = self.param('w_n', nn.initializers.lecun_normal(), (self.hidden_dim + 16, self.hidden_dim))
        b_n = self.param('b_n', nn.initializers.zeros, (self.hidden_dim,))
        def scan_fn(carry, x_t):
            combined = jnp.concatenate([carry, x_t], axis=-1)
            z = nn.sigmoid(jnp.dot(combined, w_z) + b_z)
            r = nn.sigmoid(jnp.dot(combined, w_r) + b_r)
            combined_n = jnp.concatenate([r * carry, x_t], axis=-1)
            n_state = jnp.tanh(jnp.dot(combined_n, w_n) + b_n)
            next_h = (1.0 - z) * n_state + z * carry
            return next_h, next_h
        init_h = jnp.zeros((embeds.shape[1], self.hidden_dim))
        carry, _ = jax.lax.scan(scan_fn, init_h, embeds)
        return nn.Dense(1)(carry).squeeze()

class MatNetLSTM(nn.Module):
    n: int
    @nn.compact
    def __call__(self, embeds):
        lstm_params = {
            "i": mtn.MatrixParams(self.param("i_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("i_B", nn.initializers.zeros, (2, self.n, self.n))),
            "f": mtn.MatrixParams(self.param("f_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("f_B", nn.initializers.zeros, (2, self.n, self.n))),
            "g": mtn.MatrixParams(self.param("g_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("g_B", nn.initializers.zeros, (2, self.n, self.n))),
            "o": mtn.MatrixParams(self.param("o_W", nn.initializers.lecun_normal(), (2, 3, self.n, self.n)), self.param("o_B", nn.initializers.zeros, (2, self.n, self.n)))
        }
        def batch_scan_fn(carry, x_t):
            return jax.vmap(mtn.nn.lstm_step, in_axes=(None, 0, 0))(lstm_params, carry, x_t)
        init_h = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        init_c = jnp.zeros((embeds.shape[1], 2, self.n, self.n))
        carry, _ = jax.lax.scan(batch_scan_fn, (init_h, init_c), embeds)
        return carry[0].mean(axis=(1, 2, 3))

class ScalarLSTM(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, embeds):
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
        return nn.Dense(1)(carry[0]).squeeze()

class MatNetAttention(nn.Module):
    n: int
    @nn.compact
    def __call__(self, x):
        attn_out = jax.vmap(lambda seq: mtn.lax.matrix_attention(None, seq, seq, seq))(x)
        return attn_out.mean(axis=(1, 2, 3, 4))

class ScalarAttention(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        attn_out = nn.MultiHeadDotProductAttention(num_heads=2, qkv_features=self.hidden_dim)(x, x)
        return nn.Dense(1)(attn_out.mean(axis=1)).squeeze()

def generate_task_dummy_input(task, batch_size, seq_len, input_dim, is_mat, n):
    key = jax.random.PRNGKey(0)
    if task == "TABULAR" or task == "LINEAR_REGRESSION":
        if is_mat:
            return jax.random.normal(key, (batch_size, input_dim, n, n))
        return jax.random.normal(key, (batch_size, input_dim))
    elif task == "CONV1D":
        if is_mat:
            return jax.random.normal(key, (batch_size, seq_len, 1, n, n))
        return jax.random.normal(key, (batch_size, seq_len, 1))
    elif task == "CONV2D":
        if is_mat:
            return jax.random.normal(key, (batch_size, 28, 28, 1, n, n))
        return jax.random.normal(key, (batch_size, 28, 28, 1))
    elif task == "SEQUENCE":
        if is_mat:
            return jax.random.normal(key, (seq_len, batch_size, 1, n, n))
        return jax.random.normal(key, (seq_len, batch_size, 16))

def get_loss_fn(task, state, y_batch):
    def loss_fn(p, x_batch):
        logits = state.apply_fn(p, x_batch)
        if task == "CONV2D":
            one_hot = jax.nn.one_hot(y_batch, 10)
            return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        elif task == "LINEAR_REGRESSION":
            return jnp.mean(optax.l2_loss(predictions=logits.squeeze(), targets=y_batch))
        else:
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits.squeeze(), labels=y_batch.astype(jnp.float32)))
    return loss_fn

def run_comprehensive_benchmark():
    results = []
    tasks = {
        "LINEAR_REGRESSION": (MatNetLinearRegression, ScalarLinearRegression),
        "TABULAR": (MatNetDense, ScalarDense),
        "CONV1D": (MatNetConv1D, ScalarConv1D),
        "CONV2D": (MatNetConv2D, ScalarConv2D),
        "RNN": (MatNetRNN, ScalarRNN),
        "GRU": (MatNetGRU, ScalarGRU),
        "LSTM": (MatNetLSTM, ScalarLSTM),
        "ATTENTION": (MatNetAttention, ScalarAttention)
    }

    n_values = [2, 4, 8]
    batch_sizes = [8, 16, 32]
    # Total tests per arch = 4 n * 3 batch = 12 MATNET tests + 12 Scalar tests = 24 per architecture.
    # Total tests = 24 * 8 architectures = 192 tests.

    seq_len = 16
    input_dim = 8
    key = jax.random.PRNGKey(42)

    print("Starting Grid Search for > 100 tests...")

    for task_name, (MatNetClass, ScalarClass) in tqdm(tasks.items()):
        data_category = "SEQUENCE" if task_name in ["RNN", "GRU", "LSTM", "ATTENTION"] else task_name
        if task_name == "TABULAR" or task_name == "LINEAR_REGRESSION":
            data_category = task_name

        for n in n_values:
            for batch_size in batch_sizes:
                X_train, y_train = get_dummy_data(batch_size, data_category, seq_len, input_dim)

                # Setup models
                if task_name in ["CONV1D", "CONV2D"]:
                    mat_model = MatNetClass(n=n, p=1)
                elif task_name in ["TABULAR", "LINEAR_REGRESSION"]:
                    mat_model = MatNetClass(n=n, input_dim=input_dim)
                else:
                    mat_model = MatNetClass(n=n)

                dummy_x_mat = generate_task_dummy_input(data_category, batch_size, seq_len, input_dim, True, n)
                mat_params = mat_model.init(key, dummy_x_mat)
                mat_param_count = count_params(mat_params)

                hidden_dim = 2
                while True:
                    scalar_model = ScalarClass(hidden_dim=hidden_dim)
                    dummy_x_scalar = generate_task_dummy_input(data_category, batch_size, seq_len, input_dim, False, n)
                    if task_name == "ATTENTION":
                        dummy_x_scalar = jnp.transpose(dummy_x_scalar, (1, 0, 2))
                    scalar_params = scalar_model.init(key, dummy_x_scalar)
                    if count_params(scalar_params) >= mat_param_count:
                        break
                    hidden_dim += 2

                for model_type, model, params, is_mat in [
                    ("MATNETS", mat_model, mat_params, True),
                    ("Scalar", scalar_model, scalar_params, False)
                ]:
                    tx = optax.adam(1e-3)
                    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

                    @jax.jit
                    def train_step(state, x_batch, y_batch):
                        loss_fn = get_loss_fn(task_name, state, y_batch)
                        loss, grads = jax.value_and_grad(loss_fn)(state.params, x_batch)
                        return state.apply_gradients(grads=grads), loss

                    dummy_y = jnp.array(y_train)
                    dummy_x = generate_task_dummy_input(data_category, batch_size, seq_len, input_dim, is_mat, n)
                    if task_name == "ATTENTION" and not is_mat:
                        dummy_x = jnp.transpose(dummy_x, (1, 0, 2))

                    try:
                        cost = train_step.lower(state, dummy_x, dummy_y).cost_analysis()
                        flops = cost[0].get('flops', 0) if isinstance(cost, list) and len(cost) > 0 else cost.get('flops', 0) if isinstance(cost, dict) else 0
                        state, _ = train_step(state, dummy_x, dummy_y)

                        start_time = time.time()
                        state, final_loss = train_step(state, dummy_x, dummy_y)
                        epoch_time = time.time() - start_time

                        results.append({
                            "Architecture": task_name,
                            "Model": model_type,
                            "Size_n": n if is_mat else "-",
                            "BatchSize": batch_size,
                            "Params": count_params(state.params),
                            "FLOPs": flops,
                            "Train_Time_Per_Batch": epoch_time,
                            "Final_Loss": float(final_loss)
                        })
                    except Exception as e:
                        pass

    df = pd.DataFrame(results)
    df.to_csv("summary_results_comprehensive.csv", index=False)
    print(f"Grid Search Complete! Generated {len(results)} benchmark records.")

if __name__ == "__main__":
    run_comprehensive_benchmark()
