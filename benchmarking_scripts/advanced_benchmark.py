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

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# --- MODELS ---
class MatNetConv1D(nn.Module):
    n: int
    p: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        # x input is mapped to (B, T, 1, n, n)
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

def get_loss_fn(state, y_batch, num_classes):
    def loss_fn(p, x_batch):
        logits = state.apply_fn(p, x_batch)
        if num_classes > 2:
            one_hot = jax.nn.one_hot(y_batch, num_classes)
            return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot)), logits
        else:
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits.squeeze(), labels=y_batch.astype(jnp.float32))), logits
    return loss_fn

def compute_accuracy(logits, labels, num_classes):
    if num_classes > 2:
        return jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    else:
        preds = (jax.nn.sigmoid(logits.squeeze()) > 0.5).astype(jnp.int32)
        return jnp.mean(preds == labels)

def generate_multi_data(batch_size, seq_len=64):
    datasets = []
    # Data 1: Easy pattern
    X1 = np.random.randn(batch_size, seq_len, 1).astype(np.float32)
    y1 = (X1.mean(axis=1).squeeze() > 0).astype(np.int32)
    datasets.append((X1, y1, "Data1_MeanThreshold"))

    # Data 2: Temporal variation pattern
    X2 = np.random.randn(batch_size, seq_len, 1).astype(np.float32)
    y2 = (np.diff(X2, axis=1).max(axis=1).squeeze() > 1.0).astype(np.int32)
    datasets.append((X2, y2, "Data2_DiffMax"))

    # Data 3: Random noise baseline
    X3 = np.random.randn(batch_size, seq_len, 1).astype(np.float32)
    y3 = np.random.randint(0, 2, size=(batch_size,))
    datasets.append((X3, y3, "Data3_Noise"))

    return datasets

def run_advanced_benchmark():
    n_values = [4, 8]
    batch_size = 32
    seq_len = 64
    epochs = 15
    lr = 5e-3
    num_classes = 1

    key = jax.random.PRNGKey(42)
    datasets = generate_multi_data(batch_size, seq_len)

    results = []

    for n in n_values:
        mat_model = MatNetConv1D(n=n, p=1, num_classes=num_classes)
        dummy_x = np.random.randn(batch_size, seq_len, 1).astype(np.float32)
        dummy_y = np.random.randint(0, 2, size=(batch_size,))

        mat_params = mat_model.init(key, dummy_x)
        mat_param_count = count_params(mat_params)

        # We need the JIT step to get FLOPs
        tx = optax.adam(lr)
        mat_state = TrainState.create(apply_fn=mat_model.apply, params=mat_params, tx=tx)

        @jax.jit
        def train_step_mat(state, x_batch, y_batch):
            loss_fn = get_loss_fn(state, y_batch, num_classes)
            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x_batch)
            acc = compute_accuracy(logits, y_batch, num_classes)
            return state.apply_gradients(grads=grads), loss, acc

        mat_cost = train_step_mat.lower(mat_state, dummy_x, dummy_y).cost_analysis()
        mat_flops = mat_cost[0].get('flops', 0) if isinstance(mat_cost, list) and len(mat_cost) > 0 else mat_cost.get('flops', 0) if isinstance(mat_cost, dict) else 0

        print(f"MATNETS (n={n}) | Params: {mat_param_count} | FLOPs: {mat_flops}")

        # Find Baseline A: Matching Params
        hidden_dim_params = 2
        while True:
            scalar_model_p = ScalarConv1D(hidden_dim=hidden_dim_params, num_classes=num_classes)
            scalar_params_p = scalar_model_p.init(key, dummy_x)
            if count_params(scalar_params_p) >= mat_param_count:
                break
            hidden_dim_params += 2

        # Find Baseline B: Matching FLOPs
        hidden_dim_flops = 2
        while True:
            scalar_model_f = ScalarConv1D(hidden_dim=hidden_dim_flops, num_classes=num_classes)
            scalar_params_f = scalar_model_f.init(key, dummy_x)
            scalar_state_f = TrainState.create(apply_fn=scalar_model_f.apply, params=scalar_params_f, tx=tx)

            @jax.jit
            def train_step_scalar_f(state, x_batch, y_batch):
                loss_fn = get_loss_fn(state, y_batch, num_classes)
                (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x_batch)
                return state.apply_gradients(grads=grads), loss, compute_accuracy(logits, y_batch, num_classes)

            scalar_cost_f = train_step_scalar_f.lower(scalar_state_f, dummy_x, dummy_y).cost_analysis()
            scalar_flops = scalar_cost_f[0].get('flops', 0) if isinstance(scalar_cost_f, list) and len(scalar_cost_f) > 0 else scalar_cost_f.get('flops', 0) if isinstance(scalar_cost_f, dict) else 0

            if scalar_flops >= mat_flops:
                break
            hidden_dim_flops += 2

        print(f"Matched Params Baseline | hidden_dim: {hidden_dim_params} | Params: {count_params(scalar_params_p)}")
        print(f"Matched FLOPs Baseline  | hidden_dim: {hidden_dim_flops} | FLOPs: {scalar_flops} | Params: {count_params(scalar_params_f)}")

        models_to_test = [
            ("MATNETS", mat_model, mat_params),
            ("Scalar_MatchParams", scalar_model_p, scalar_params_p),
            ("Scalar_MatchFLOPs", scalar_model_f, scalar_params_f)
        ]

        for X_data, y_data, data_name in datasets:
            for model_type, model, params in models_to_test:
                state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

                @jax.jit
                def train_step(state, x_batch, y_batch):
                    loss_fn = get_loss_fn(state, y_batch, num_classes)
                    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x_batch)
                    acc = compute_accuracy(logits, y_batch, num_classes)
                    return state.apply_gradients(grads=grads), loss, acc

                # We record accuracy over epochs to see learning speed
                epoch_accuracies = []
                for ep in range(epochs):
                    state, l, acc = train_step(state, X_data, y_data)
                    epoch_accuracies.append(float(acc))

                # Re-calculate exact flops for logging
                cost = train_step.lower(state, X_data, y_data).cost_analysis()
                actual_flops = cost[0].get('flops', 0) if isinstance(cost, list) and len(cost) > 0 else cost.get('flops', 0) if isinstance(cost, dict) else 0

                results.append({
                    "Dataset": data_name,
                    "Model": model_type,
                    "Size_n": n if model_type == "MATNETS" else "-",
                    "Params": count_params(state.params),
                    "FLOPs": actual_flops,
                    "Epoch_5_Acc": epoch_accuracies[4] if len(epoch_accuracies) >= 5 else 0,
                    "Epoch_10_Acc": epoch_accuracies[9] if len(epoch_accuracies) >= 10 else 0,
                    "Final_Acc": epoch_accuracies[-1]
                })

    df = pd.DataFrame(results)
    df.to_csv("advanced_benchmark_results.csv", index=False)
    print("Advanced Benchmarking Complete. Results saved to advanced_benchmark_results.csv")

if __name__ == "__main__":
    run_advanced_benchmark()
