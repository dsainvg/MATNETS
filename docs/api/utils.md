# `matnets.utils`

Data preprocessing utilities for MATNETS.

```python
from matnets.utils import embed_pixels
import numpy as np

imgs = np.zeros((2, 10, 10, 3))  # (Batch, H, W, Channels)
windows = embed_pixels(imgs, n=3, spatial_axes=(1, 2), interleave=False)
# Shape: (2, 10, 10, 3, 3, 3)
```

`embed_pixels` extracts an `n x n` (or `n` for 1D) local neighborhood around
each element. The function automatically applies zero padding so the output
spatial dimensions match the input spatial dimensions, with the new window
dimensions appended to the end of the shape.

If `interleave=True` (or a tuple of booleans per axis), the order of elements
along the spatial axes is permuted according to an interleaved block pattern.

### `embed_sequence`

```python
from matnets.utils import embed_sequence
import numpy as np

seq = np.array([1, 2, 3, 4, 5])  # 1D Sequence: (T,)
out = embed_sequence(seq, n=3, axis=0)
# Shape: (5, 3, 3)

# For multiple channels / batch:
seq_mc = np.zeros((10, 5)) # (T, C)
out_mc = embed_sequence(seq_mc, n=3, axis=0)
# Shape: (10, 5, 3, 3)
```

`embed_sequence` extracts a symmetric `n x n` time-history embedding over a given time axis, ideal for audio, time-series, or other sequentially streaming data.

For every time step `t` along the target sequence `axis`, this backwardly extracts history up to `n` steps, constructing a symmetric matrix where distance from the diagonal corresponds naturally to the delay. Previous states prior to `t=0` are strictly zero-padded.

It supports native mapping over 1D, 2D, and 3D data formats (e.g. `(T,)`, `(N, T)`, `(N, T, C)`) without interfering with non-sequential dimensions.
