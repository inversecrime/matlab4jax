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

# Installation
Pre-compiled wheels are available for MATLAB R2026a with Python 3.12 and Python 3.13.

If you are using a supported configuration, install with:
```bash
pip install matlab4jax
```
Otherwise, install from source:
```bash
pip install --no-binary matlab4jax matlab4jax
```

## Shape convention
MATLAB and JAX use different conventions for array shapes. While JAX supports 0-dimensional and 1-dimensional arrays, in MATLAB every array has at least two dimensions. Additionally, MATLAB hides trailing singleton dimensions. The following conversions will be applied:
| JAX shape  | MATLAB shape |
| ---------- | ------------ |
| `()`       | `(1, 1)`     |
| `(n,)`     | `(1, n)`     |
| `(..., 1)` | `(...)`      |

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
