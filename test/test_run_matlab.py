import itertools
from math import prod

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

import matlab4jax

jax.config.update("jax_enable_x64", True)

key = jax.random.key(0)

for (shape, dtype) in itertools.product([(7, 1), (10,), (1, 7, 3, 1, 23), (3000, 4000, 3), (3, 4, 5, 6)], [jnp.float32, jnp.float64]):
    (key, subkey) = jax.random.split(key)
    x = jax.random.normal(subkey, shape=shape, dtype=dtype)

    y_jax = x * x
    z_jax = jnp.reshape(x, shape=(1, -1), order="F")

    [y_matlab, z_matlab] = matlab4jax.run_and_time_matlab(
        inputs=[x],
        input_names=["x"],
        command="y = x .* x; z = reshape(x, 1, []);",
        output_names=["y", "z"],
        abstract_outputs=[ShapeDtypeStruct(y_jax.shape, y_jax.dtype), ShapeDtypeStruct(z_jax.shape, z_jax.dtype)],
        use_file=True,
    )

    assert jnp.allclose(y_jax, y_matlab)
    assert jnp.allclose(z_jax, z_matlab)

print(jax.make_jaxpr(lambda: matlab4jax.run_matlab(
    inputs=[x],
    input_names=["x"],
    command="y = x .* x; z = reshape(x, 1, []);",
    output_names=["y", "z"],
    abstract_outputs=[ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), ShapeDtypeStruct(shape=(1, prod(x.shape)), dtype=x.dtype)],
    use_file=True,
))())
