# matlab4jax
Run MATLAB code from JAX with `jax.jit` support.

## Example
```python
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

import matlab4jax

x = jnp.array([[1.0, 2.0],
               [3.0, 4.0]])

[y] = jax.jit(lambda x: matlab4jax.run_matlab(
    inputs=[x],
    input_names=["x"],
    command="y = inv(x);",
    output_names=["y"],
    abstract_outputs=[ShapeDtypeStruct(x.shape, x.dtype)]
))(x)

print(y)
```
Output:
```
[[-2.0000002   1.0000001 ]
 [ 1.5000001  -0.50000006]]
```

## Installation
Requires:
- JAX
- MATLAB with the MATLAB Engine API
- C++ compiler and CMake

Use `pip install .` to install `matlab4jax`.

## License
Licensed under the [MIT License](LICENSE.md).

## Matlab library path
On Linux, set the Matlab library path before importing, for example:
```bash
export LD_LIBRARY_PATH=/usr/local/MATLAB/R2026a/extern/bin/glnxa64:$LD_LIBRARY_PATH
```
To make this setting persistent, add it to your `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/MATLAB/R2026a/extern/bin/glnxa64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
