import timeit
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array, ShapeDtypeStruct

import matlab4jax
import matlab4jax.matlabengine


def with_jax(x: Array) -> Array:
    [y] = matlab4jax.eval_function("inv", [x])
    return y


def with_matlabengine(x: Array) -> Array:
    [y] = matlab4jax.matlabengine.eval_function("inv", [x])
    return y


def with_xla(x: Array, *, use_file: bool) -> Array:
    [y] = matlab4jax.run_matlab(
        inputs=[x],
        input_names=["x"],
        command="y = inv(x);",
        output_names=["y"],
        abstract_outputs=[ShapeDtypeStruct(x.shape, x.dtype)],
        use_file=use_file,
    )
    return y


def benchmark(function: Callable[..., Any]) -> None:
    repeat = 5
    number = 3
    times = timeit.repeat(lambda: jax.block_until_ready(function(x)), repeat=repeat, number=number)
    print(f"mean: {sum(times) / len(times) / number:.3f}s")
    print(f"min:  {min(times) / number:.3f}s")


n = 1300
x = jax.random.normal(jax.random.key(0), shape=(n, n))
jax.block_until_ready(x)
y1 = with_jax(x)
y2 = with_matlabengine(x)
y3 = with_xla(x, use_file=True)
y4 = with_xla(x, use_file=False)
assert jnp.array_equiv(y1, y2)
assert jnp.array_equiv(y1, y3)
assert jnp.array_equiv(y1, y4)

print("=== jax ===")
benchmark(with_jax)
print("=== matlabengine ===")
benchmark(with_matlabengine)
print("=== xla, use_file=True ===")
benchmark(partial(with_xla, use_file=True))
print("=== xla, use_file=False ===")
benchmark(partial(with_xla, use_file=False))
print("=== xla, use_file=True, jit ===")
benchmark(jax.jit(partial(with_xla, use_file=True)))
print("=== xla, use_file=False, jit ===")
benchmark(jax.jit(partial(with_xla, use_file=False)))
