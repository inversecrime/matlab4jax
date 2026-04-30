import time
from typing import Sequence

import jax._src.core
import jax._src.pretty_printer
import jax.extend
import jax.interpreters.mlir
import jax.numpy as jnp
import matlab4jax.matlab4jax_cpp
import numpy as np
from jax import Array, ShapeDtypeStruct
from jax.extend.core import Primitive
from jax.core import ShapedArray, Tracer

# Alternative without cpp:
# import matlab.engine
# engine = matlab.engine.start_matlab("matlab4jax")
# def run_matlab_impl(
#     *inputs: Array,
#     input_variable_names: tuple[str,...],
#     command: str,
#     output_variable_names: tuple[str,...],
#     abstract_outputs: tuple[ShapeDtypeStruct,...]
# ) -> tuple[Array,...]:
#     def callback(inputs: tuple[Array,...]) -> tuple[np,....ndarray]:
#         for (name, value) in zip(input_variable_names, inputs):
#             engine.workspace[name] = np.asarray(value)  # type: ignore
#         engine.eval(command, nargout=0)  # type: ignore
#         outputs = [np.asarray(engine.workspace[name]) for name in output_variable_names]  # type: ignore
#         outputs = [np.expand_dims(value, axis=tuple(range(value.ndim, 2))) for value in outputs]
#         outputs = [np.squeeze(value, axis=tuple(range(2 - abstract_value.ndim))) for (value, abstract_value) in zip(outputs, abstract_outputs)]
#         return outputs
#     return jax.pure_callback(callback, abstract_outputs, inputs)

jax.ffi.register_ffi_target("run_matlab", matlab4jax.matlab4jax_cpp.run_matlab(), platform="cpu")


def encode_string(string: str) -> np.ndarray:
    byte_data = string.encode("utf8")
    return np.frombuffer(byte_data, dtype=np.uint8)


def encode_string_list(string_list: tuple[str, ...]) -> np.ndarray:
    byte_data = bytearray()
    for string in string_list:
        encoded_string = string.encode("utf8")
        byte_data.extend(len(encoded_string).to_bytes(4, byteorder="little", signed=False))
        byte_data.extend(encoded_string)
    return np.frombuffer(byte_data, dtype=np.uint8)


def run_matlab_impl(
    *inputs: Array,
    input_variable_names: tuple[str, ...],
    command: str,
    output_variable_names: tuple[str, ...],
    abstract_outputs: tuple[ShapeDtypeStruct, ...]
) -> list[Array]:
    return list(jax.ffi.ffi_call(
        "run_matlab",
        abstract_outputs,
        input_layouts=[tuple(reversed(range(input.ndim))) for input in inputs],
        output_layouts=[tuple(reversed(range(abstract_output.ndim))) for abstract_output in abstract_outputs]
    )(
        *inputs,
        command_as_bytes=encode_string(command),
        input_variable_names_as_bytes=encode_string_list(input_variable_names),
        output_variable_names_as_bytes=encode_string_list(output_variable_names)
    ))


def run_matlab_abstract_eval(
    *inputs: ShapedArray,
    input_variable_names: tuple[str, ...],
    command: str,
    output_variable_names: tuple[str, ...],
    abstract_outputs: tuple[ShapeDtypeStruct, ...]
) -> list[ShapedArray]:
    return [ShapedArray(shape=abstract_output.shape, dtype=abstract_output.dtype) for abstract_output in abstract_outputs]


def run_matlab_pp_rule(eqn: jax._src.core.JaxprEqn, context: jax._src.core.JaxprPpContext, settings: jax._src.core.JaxprPpSettings) -> jax._src.pretty_printer.Doc:
    return jax._src.core._pp_eqn(eqn, context, settings, ["input_variable_names", "command", "output_variable_names"])


def run_matlab(
    inputs: Sequence[Array],
    input_variable_names: Sequence[str],
    command: str,
    output_variable_names: Sequence[str],
    abstract_outputs: Sequence[ShapeDtypeStruct]
) -> list[Array]:
    return list(run_matlab_p.bind(
        *inputs,
        input_variable_names=tuple(input_variable_names),
        command=command,
        output_variable_names=tuple(output_variable_names),
        abstract_outputs=tuple(abstract_outputs)
    ))


def run_and_time_matlab(
    inputs: Sequence[Array],
    input_variable_names: Sequence[str],
    command: str,
    output_variable_names: Sequence[str],
    abstract_outputs: Sequence[ShapeDtypeStruct]
) -> list[Array]:
    if isinstance(jnp.array(0), Tracer):
        raise Exception("Cannot use \"run_and_time_matlab\" inside of transformations.")

    overall_runtime = time.time()

    runtime_variable_name = "G7B2KX9M4J1Z5T8W3Q6Y"
    command = f"{runtime_variable_name} = tic;\n{command}\n{runtime_variable_name} = toc({runtime_variable_name});"
    output_variable_names = [runtime_variable_name, *output_variable_names]
    abstract_outputs = [ShapeDtypeStruct(shape=(1, 1), dtype=jnp.float64), *abstract_outputs]
    [runtime, *outputs] = run_matlab(
        inputs=inputs,
        input_variable_names=input_variable_names,
        command=command,
        output_variable_names=output_variable_names,
        abstract_outputs=abstract_outputs
    )
    runtime = jnp.squeeze(runtime)
    jax.block_until_ready([runtime, outputs])

    overall_runtime = time.time() - overall_runtime
    overhead = overall_runtime - runtime

    print(f"matlab runtime: {runtime:.2f}")
    print(f"overhead:       {overhead:.2f}")

    return outputs


run_matlab_p = Primitive("run_matlab")
run_matlab_p.multiple_results = True
run_matlab_p.def_impl(run_matlab_impl)
run_matlab_p.def_abstract_eval(run_matlab_abstract_eval)
jax._src.core.pp_eqn_rules[run_matlab_p] = run_matlab_pp_rule
jax.interpreters.mlir.register_lowering(run_matlab_p, jax.interpreters.mlir.lower_fun(run_matlab_impl, multiple_results=True))
