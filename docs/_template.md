# [Feature/Concept Name]

## Overview

[Provide a concise explanation of what the feature or concept is, what problem it solves, and when to use it in MATNETS.]

## Code Example

[Every feature or concept **must** include a practical, well-commented code snippet or example. The example should clearly demonstrate how to initialize, configure, and run the feature.]

```python
import jax
import jax.numpy as jnp
import matnets as mtn

# Initialize parameters
# ...

# Prepare input data with correct matrix-neuron shapes (e.g. p, n, n)
# ...

# Apply the operation
# ...

# Expected Output Shape: (...)
```

## Best Practices

- **Rule 1:** [E.g., Ensure the matrix dimension `n` matches across all inputs.]
- **Rule 2:** [E.g., Batch dimensions should be handled via `jax.vmap` rather than manual loops.]

## Related API

- `mtn.some_function`
- `mtn.another_feature`
