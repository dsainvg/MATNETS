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

print("Starting CNN Benchmark Data Loading...")

def get_mnist_data(batch_size):
    dataset = load_dataset('mnist')
    # Use smaller subset for faster benchmarking if needed, but let's use a decent size to see actual per-epoch times.
    # To run relatively fast, let's use 10,000 for train, 1,000 for test
    train_data = dataset['train'].select(range(200))
    test_data = dataset['test'].select(range(200))

    X_train = np.array(train_data['image']).astype(np.float32) / 255.0
    y_train = np.array(train_data['label'])

    X_test = np.array(test_data['image']).astype(np.float32) / 255.0
    y_test = np.array(test_data['label'])

    # Reshape for scalar CNN: (N, H, W, 1) - actually CNN2D expects (N, H, W, C) where C is channel
    X_train_scalar = np.expand_dims(X_train, axis=-1)
    X_test_scalar = np.expand_dims(X_test, axis=-1)

    # We will generate MATNETS shape data dynamically inside the training loop to match p, n

    return (X_train_scalar, y_train), (X_test_scalar, y_test)


class ScalarCNN(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # x is (B, H, W, 1)
        x = nn.Conv(features=self.hidden_dim, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=self.hidden_dim * 2, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

class MatNetCNN(nn.Module):
    p: int
    q1: int
    q2: int
    n: int

    @nn.compact
    def __call__(self, img):
        # img shape is (B, H, W, p, n, n)

        c1 = mtn.MatrixParams(
            W=self.param("c1_W", nn.initializers.lecun_normal(), (self.q1, self.p, 3, 3, self.n, self.n)),
            B=self.param("c1_B", nn.initializers.zeros, (self.q1, self.n, self.n)),
        )
        c2 = mtn.MatrixParams(
            W=self.param("c2_W", nn.initializers.lecun_normal(), (self.q2, self.q1, 3, 3, self.n, self.n)),
            B=self.param("c2_B", nn.initializers.zeros, (self.q2, self.n, self.n)),
        )
        outp = mtn.MatrixParams(
            W=self.param("out_W", nn.initializers.lecun_normal(), (10, self.q2, self.n, self.n)),
            B=self.param("out_B", nn.initializers.zeros, (10, self.n, self.n)),
        )

        h = jax.nn.relu(jax.vmap(lambda p, x: mtn.lax.matrix_conv2d(p, x, padding="SAME"), in_axes=(None, 0))(c1, img))
        # Downsample with stride
        h = jax.nn.relu(jax.vmap(lambda p, x: mtn.lax.matrix_conv2d(p, x, stride=(2, 2), padding="SAME"), in_axes=(None, 0))(c2, h))

        # Spatial global average pooling
        h = h.mean(axis=(1, 2))

        # Dense classification layer
        out = jax.vmap(mtn.dense, in_axes=(None, 0))(outp, h)

        # Final aggregation over matrix dims to get logits (B, 10)
        return out.mean(axis=(2, 3))


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def benchmark_cnn():
    batch_size = 64
    epochs = 1

    (X_train, y_train), (X_test, y_test) = get_mnist_data(batch_size)
    num_batches = len(X_train) // batch_size

    results = []

    for n in [8, 16]:
        # MATNETS config
        p = 2
        q1 = 4
        q2 = 8

        # Calculate MATNETS params
        # To just initialize we can create dummy data
        key = jax.random.PRNGKey(0)
        dummy_img_mat = jax.random.normal(key, (1, 28, 28, p, n, n))
        mat_model = MatNetCNN(p=p, q1=q1, q2=q2, n=n)
        mat_params = mat_model.init(key, dummy_img_mat)
        mat_param_count = count_params(mat_params)

        print(f"MATNETS (n={n}) Param Count: {mat_param_count}")

        # Find comparable scalar hidden_dim
        # Rough heuristic: scalar hidden dim proportional to matnets channels * n
        hidden_dim = 16
        while True:
            scalar_model = ScalarCNN(hidden_dim=hidden_dim)
            scalar_params = scalar_model.init(key, jax.random.normal(key, (1, 28, 28, 1)))
            scalar_param_count = count_params(scalar_params)
            if scalar_param_count >= mat_param_count:
                break
            hidden_dim += 2

        print(f"Scalar Baseline (hidden_dim={hidden_dim}) Param Count: {scalar_param_count}")

        # Benchmarking loop for both models
        for model_type, model, params, is_mat in [
            ("MATNETS", mat_model, mat_params, True),
            ("ScalarBaseline", scalar_model, scalar_params, False)
        ]:
            tx = optax.adam(1e-3)
            state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

            @jax.jit
            def train_step(state, x_batch, y_batch):
                def loss_fn(p):
                    logits = state.apply_fn(p, x_batch)
                    one_hot = jax.nn.one_hot(y_batch, 10)
                    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
                    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y_batch)
                    return loss, acc
                (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                state = state.apply_gradients(grads=grads)
                return state, loss, acc

            # Lower & FLOPs cost analysis
            if is_mat:
                dummy_x = jax.random.normal(key, (batch_size, 28, 28, p, n, n))
            else:
                dummy_x = jax.random.normal(key, (batch_size, 28, 28, 1))
            dummy_y = jax.random.randint(key, (batch_size,), 0, 10)

            cost = train_step.lower(state, dummy_x, dummy_y).cost_analysis()
            flops = cost[0].get('flops', 0) if isinstance(cost, list) and len(cost) > 0 else cost.get('flops', 0) if isinstance(cost, dict) else 0

            # Warmup
            state, _, _ = train_step(state, dummy_x, dummy_y)

            total_time = 0
            for epoch in range(epochs):
                # Shuffle
                perm = np.random.permutation(len(X_train))
                X_train_shuf = X_train[perm]
                y_train_shuf = y_train[perm]

                start_time = time.time()
                epoch_loss = 0
                epoch_acc = 0

                for i in range(num_batches):
                    batch_x = X_train_shuf[i*batch_size:(i+1)*batch_size]
                    batch_y = y_train_shuf[i*batch_size:(i+1)*batch_size]

                    if is_mat:
                        # Map scalar (B, 28, 28, 1) -> (B, 28, 28, p, n, n)
                        # We duplicate/broadcast the scalar value into the matrix diagonal or just fill
                        # For simplicity, just tile to fill (p, n, n)
                        batch_x = np.tile(batch_x[..., np.newaxis, np.newaxis], (1, 1, 1, p, n, n))

                    state, loss, acc = train_step(state, batch_x, batch_y)
                    epoch_loss += loss
                    epoch_acc += acc

                # synchronize
                epoch_loss = epoch_loss.item() / num_batches
                epoch_acc = epoch_acc.item() / num_batches

                epoch_time = time.time() - start_time
                total_time += epoch_time

            results.append({
                "Task": "MNIST CNN",
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
    print("CNN Benchmarks saved to summary_results.csv")

if __name__ == "__main__":
    benchmark_cnn()
