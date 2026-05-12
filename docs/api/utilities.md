# Utilities

### `matnets.utils.embed_pixels`

A preprocessing utility to convert standard image tensors into overlapping matrix-valued neighborhoods.

```python
from matnets.utils import embed_pixels
import numpy as np

# A standard image batch: (Batch, H, W, Channels)
imgs = np.zeros((2, 10, 10, 3))

# Extract 3x3 local neighborhoods
windows = embed_pixels(imgs, n=3, spatial_axes=(1, 2), interleave=False)

# Shape: (2, 10, 10, 3, 3, 3)
# -> (Batch, H, W, Channels, n, n)
```

**Arguments:**

- `imgs` (np.ndarray | jax.Array): The input image tensor.
- `n` (int): The window size (will become the $n \times n$ matrix dimensions).
- `spatial_axes` (tuple): The axes corresponding to height and width.
- `interleave` (bool | tuple): If true, permutes the order of elements along the spatial axes.
