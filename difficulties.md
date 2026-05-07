# API Friction and Difficulties Documented

During the benchmarking process of MATNETS, a few points of friction and limitations were identified within the high-level API.

## 1. Batch Dimension Mapping (vmap Requirements)
High-level MATNETS primitives like `mtn.lax.matrix_conv2d` and `mtn.dense` are strictly defined over single examples (e.g., shape `(y, x, p, n, n)`).
While standard Flax/JAX modules (like `nn.Conv`) natively handle the batch dimension by default, using MATNETS inside a Flax `nn.Module` requires explicit `jax.vmap` wrapping around every primitive.
* **Friction**: This breaks standard Flax composability. Users expect `mtn.lax.matrix_conv2d` to act as a drop-in replacement, but instead, they must write custom `vmap` wrappers inside `@nn.compact` scopes.

## 2. Input Shape Transitions
When converting data from standard datasets (like MNIST `(H, W, 1)`) to MATNETS `(H, W, p, n, n)`, there is no built-in "projection" primitive.
* **Friction**: Users have to manually tile or broadcast scalar values across matrices, which may not be mathematically sound or efficient. An explicit `mtn.scalar_to_matrix(x)` or an embedding layer primitive is highly recommended.

## 3. Lacking standard Pooling Primitives
The `matrix_conv2d` supports `stride`, which allows for downsampling, but the library completely lacks pooling operations (like matrix max-pool or average-pool).
* **Friction**: Users must resort to raw `jnp.mean` operations across spatial axes, which might destroy the matrix semantics encoded by the library.

## 4. LSTM Integration
`mtn.nn.lstm_step` requires a dictionary of parameters `{"i", "f", "g", "o"}` and returns `((h, c), h)`.
* **Friction**: It does not easily interface with standard JAX `scan` without custom wrapping, and the dictionary parameter signature deviates from standard Flax module parameter trees, making parameter initialization verbose.
