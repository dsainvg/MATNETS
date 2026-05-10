import jax
import jax.numpy as jnp
import matnets as mtn
from matnets.activations import relu, relud, elud

def main():
    # Setup some test matrices
    # 1. Identity (det = 1.0 > 0)
    # 2. Swapped identity (det = -1.0 < 0)
    # 3. Singular (det = 0.0)
    x = jnp.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ])
    
    print("Input Matrices:")
    print(x)
    print("\nDeterminants:")
    print(jnp.linalg.det(x))

    # 1. Element-wise ReLU
    print("\n--- Element-wise ReLU (Standard) ---")
    print("Applied to all elements independently.")
    print(relu(x))

    # 2. Determinant-gated ReLU (relud)
    print("\n--- Determinant-gated ReLU (relud) ---")
    print("Zeros out the whole matrix if det <= 0.")
    print(relud(x))

    # 3. Determinant-gated ELU (elud)
    print("\n--- Determinant-gated ELU (elud) ---")
    print("Branches to alpha * (expm(X) - I) if det <= 0.")
    print(elud(x))

    # Usage in a dense layer
    print("\n--- Usage in a Dense Layer ---")
    params = mtn.init(jax.random.key(0), p=3, q=2, n=2)
    
    # Passing activation as an argument
    output = mtn.dense(params, x, activation=relud)
    print(f"Output shape with relud: {output.shape}")
    print("Output matrices:")
    print(output)

if __name__ == "__main__":
    main()
