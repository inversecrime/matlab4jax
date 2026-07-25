import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

import matlab4jax

x = jnp.array([[1., 2.],
               [3., 4.]])

[y] = jax.jit(lambda x: matlab4jax.run_matlab(
    inputs=[x],
    input_names=["x"],
    command="y = inv(x);",
    output_names=["y"],
    abstract_outputs=[ShapeDtypeStruct(x.shape, x.dtype)]
))(x)

print(y)
