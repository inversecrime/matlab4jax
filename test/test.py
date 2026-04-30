import itertools
from math import prod

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

import matlab4jax

jax.config.update("jax_enable_x64", True)

key = jax.random.key(0)

for (shape, dtype) in itertools.product(
    [(7, 1), (1, 10), (1, 7, 3, 1, 23), (3000, 4000, 3), (3, 4, 5, 6)],
    [jnp.float32, jnp.float64]
):
    (key, subkey) = jax.random.split(key)
    x = jax.random.normal(subkey, shape=shape, dtype=dtype)

    [y, z] = matlab4jax.run_and_time_matlab(
        inputs=[x],
        input_variable_names=["x"],
        command="y = x .* x; z = reshape(x, 1, []);",
        output_variable_names=["y", "z"],
        abstract_outputs=[ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), ShapeDtypeStruct(shape=(1, prod(x.shape)), dtype=x.dtype)]
    )

    assert jnp.array_equal(y, x * x)
    assert jnp.array_equal(z, jnp.reshape(x, shape=(1, prod(x.shape)), order="F"))

print(jax.make_jaxpr(lambda: matlab4jax.run_matlab(
    inputs=[x],
    input_variable_names=["x"],
    command="y = x .* x; z = reshape(x, 1, []);",
    output_variable_names=["y", "z"],
    abstract_outputs=[ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), ShapeDtypeStruct(shape=(1, prod(x.shape)), dtype=x.dtype)]
))())
