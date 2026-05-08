import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matnets as mtn
import numpy as np
import time
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_digits
from sklearn.preprocessing import StandardScaler

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# --- ARCHITECTURES ---
# Both models will have exactly 2 hidden layers to match the structure.

class MatNetDense(nn.Module):
    n: int
    input_dim: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        # Embed tabular row into matrix (B, input_dim, n, n)
        x_mapped = jnp.tile(x[..., None, None], (1, 1, self.n, self.n))

        # Hidden Layer 1 (Matrix-Neuron Count = 4)
        w1 = self.param("W1", nn.initializers.lecun_normal(), (4, self.input_dim, self.n, self.n))
        b1 = self.param("B1", nn.initializers.zeros, (4, self.n, self.n))
        h1 = jax.nn.relu(jax.vmap(lambda x_t: mtn.dense(mtn.MatrixParams(W=w1, B=b1), x_t))(x_mapped))

        # Hidden Layer 2 (Matrix-Neuron Count = 4)
        w2 = self.param("W2", nn.initializers.lecun_normal(), (4, 4, self.n, self.n))
        b2 = self.param("B2", nn.initializers.zeros, (4, self.n, self.n))
        h2 = jax.nn.relu(jax.vmap(lambda h_t: mtn.dense(mtn.MatrixParams(W=w2, B=b2), h_t))(h1))

        # Output Layer
        w3 = self.param("W3", nn.initializers.lecun_normal(), (self.num_classes, 4, self.n, self.n))
        b3 = self.param("B3", nn.initializers.zeros, (self.num_classes, self.n, self.n))
        out = jax.vmap(lambda h_t: mtn.dense(mtn.MatrixParams(W=w3, B=b3), h_t))(h2)

        # Reduce matrix output to scalar logits
        out = out.mean(axis=(2, 3))
        return out.squeeze() if self.num_classes == 1 else out

class ScalarDense(nn.Module):
    hidden_dim: int
    num_classes: int
    @nn.compact
    def __call__(self, x):
        # Hidden Layer 1
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Hidden Layer 2
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        # Output Layer
        x = nn.Dense(self.num_classes)(x)
        return x.squeeze() if self.num_classes == 1 else x

def get_loss_fn(task_type, state, y_batch, num_classes):
    def loss_fn(p, x_batch):
        logits = state.apply_fn(p, x_batch)
        if task_type == "REGRESSION":
            return jnp.mean(optax.l2_loss(predictions=logits.squeeze(), targets=y_batch)), logits
        elif task_type == "BINARY_CLASSIFICATION":
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits.squeeze(), labels=y_batch.astype(jnp.float32))), logits
        elif task_type == "MULTICLASS_CLASSIFICATION":
            one_hot = jax.nn.one_hot(y_batch, num_classes)
            return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot)), logits
    return loss_fn

def compute_accuracy(task_type, logits, labels):
    if task_type == "REGRESSION":
        # Using MSE as the proxy for "accuracy" in regression
        return jnp.mean(jnp.square(logits.squeeze() - labels))
    elif task_type == "BINARY_CLASSIFICATION":
        preds = (jax.nn.sigmoid(logits.squeeze()) > 0.5).astype(jnp.int32)
        return jnp.mean(preds == labels)
    elif task_type == "MULTICLASS_CLASSIFICATION":
        return jnp.mean(jnp.argmax(logits, axis=-1) == labels)

def load_all_datasets():
    datasets = {}

    # California Housing (Regression)
    data = fetch_california_housing()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.float32)
    datasets['California_Housing'] = {"X": X, "y": y, "type": "REGRESSION", "classes": 1, "input_dim": X.shape[1]}

    # Diabetes (Regression)
    data = load_diabetes()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.float32)
    datasets['Diabetes'] = {"X": X, "y": y, "type": "REGRESSION", "classes": 1, "input_dim": X.shape[1]}

    # Breast Cancer (Binary)
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.int32)
    datasets['Breast_Cancer'] = {"X": X, "y": y, "type": "BINARY_CLASSIFICATION", "classes": 1, "input_dim": X.shape[1]}

    # Digits (Multiclass)
    data = load_digits()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.int32)
    datasets['Digits_Dataset'] = {"X": X, "y": y, "type": "MULTICLASS_CLASSIFICATION", "classes": 10, "input_dim": X.shape[1]}

    return datasets

def run_dataset_benchmark():
    results = []
    datasets = load_all_datasets()

    n = 8
    batch_size = 64
    epochs = 20
    lr = 1e-3
    key = jax.random.PRNGKey(42)

    print("Starting identical architecture tests across diverse datasets...")

    for name, data_info in datasets.items():
        X_full = data_info["X"]
        y_full = data_info["y"]
        task_type = data_info["type"]
        num_classes = data_info["classes"]
        input_dim = data_info["input_dim"]

        num_batches = len(X_full) // batch_size
        if num_batches == 0: continue

        # 1. Initialize MATNETS Model
        mat_model = MatNetDense(n=n, input_dim=input_dim, num_classes=num_classes)
        dummy_x = X_full[:batch_size]
        mat_params = mat_model.init(key, dummy_x)
        mat_param_count = count_params(mat_params)

        # 2. Match Scalar Baseline Model Parameters
        hidden_dim = 2
        while True:
            scalar_model = ScalarDense(hidden_dim=hidden_dim, num_classes=num_classes)
            scalar_params = scalar_model.init(key, dummy_x)
            if count_params(scalar_params) >= mat_param_count:
                break
            hidden_dim += 2

        print(f"\n--- {name} ({task_type}) ---")
        print(f"MATNETS Params: {mat_param_count} | Scalar Params: {count_params(scalar_params)} (Hidden Dim: {hidden_dim})")

        # 3. Train Both Models
        for model_type, model, params in [
            ("MATNETS", mat_model, mat_params),
            ("Scalar_MatchParams", scalar_model, scalar_params)
        ]:
            tx = optax.adam(lr)
            state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

            @jax.jit
            def train_step(state, x_batch, y_batch):
                loss_fn = get_loss_fn(task_type, state, y_batch, num_classes)
                (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x_batch)
                acc = compute_accuracy(task_type, logits, y_batch)
                return state.apply_gradients(grads=grads), loss, acc

            # Epoch loop
            final_loss = 0
            final_acc = 0
            for ep in range(epochs):
                # Shuffle
                perm = np.random.permutation(len(X_full))
                X_shuf = X_full[perm]
                y_shuf = y_full[perm]

                epoch_loss = 0
                epoch_acc = 0
                for i in range(num_batches):
                    bx = X_shuf[i*batch_size:(i+1)*batch_size]
                    by = y_shuf[i*batch_size:(i+1)*batch_size]
                    state, l, acc = train_step(state, bx, by)
                    epoch_loss += l
                    epoch_acc += acc

                final_loss = epoch_loss / num_batches
                final_acc = epoch_acc / num_batches

            results.append({
                "Dataset": name,
                "Task_Type": task_type,
                "Model": model_type,
                "Hidden_Layers": 2,
                "Params": count_params(state.params),
                "Final_Loss": float(final_loss),
                "Final_Performance": float(final_acc) # Accuracy for Classification, MSE for Regression
            })
            perf_metric = "MSE" if task_type == "REGRESSION" else "Accuracy"
            print(f"  {model_type} -> Loss: {final_loss:.4f} | {perf_metric}: {final_acc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("dataset_benchmark_results.csv", index=False)
    print("\nBenchmark saved to dataset_benchmark_results.csv")

if __name__ == "__main__":
    run_dataset_benchmark()
