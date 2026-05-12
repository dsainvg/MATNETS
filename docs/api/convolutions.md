# Convolutions

MATNETS extends matrix-neurons to spatial data via matrix-based convolutions.

=== "1D Convolution"

    ```python
    from matnets.lax import matrix_conv1d

    y = matrix_conv1d(params, x, stride=1, padding="VALID")
    ```

    **Expected Shapes:**

    - `params.W`: `(q, p, kernel_size, n, n)`
    - `params.B`: `(q, n, n)`
    - `x`: `(seq_len, p, n, n)`
    - `y`: `(out_seq_len, q, n, n)`

=== "2D Convolution"

    ```python
    from matnets.lax import matrix_conv2d

    y = matrix_conv2d(params, x, stride=(1, 1), padding="SAME")
    ```

    **Expected Shapes:**

    - `params.W`: `(q, p, height, width, n, n)`
    - `params.B`: `(q, n, n)`
    - `x`: `(h_in, w_in, p, n, n)`
    - `y`: `(h_out, w_out, q, n, n)`
