import time
from pathlib import Path
from typing import Sequence

import jax._src.core
import jax._src.pretty_printer
import jax.interpreters.mlir
import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.typing import ArrayLike

import matlab4jax.matlab4jax_cpp

jax.default_device = jax.devices("cpu")[0]


def eval_command(command: str, /) ->None:
    matlab4jax.matlab4jax_cpp.eval_command(command)


def set_variable(name: str, value: ArrayLike, /) ->None:
    matlab4jax.matlab4jax_cpp.set_variable(name, jnp.asarray(value))


def get_variable(name: str, /) ->Array:
    return matlab4jax.matlab4jax_cpp.get_variable(name)


def eval_function(name_or_path: str | Path, inputs: Sequence[ArrayLike], /, *, n_outputs: int | None = None) -> list[Array]:
    eval_command("clear;")
    if isinstance(name_or_path, Path) or name_or_path.endswith(".m"):
        name_or_path = Path(name_or_path)
        assert name_or_path.exists()
        eval_command(f"addpath('{name_or_path.parent}');")
        name_or_path = name_or_path.stem
    if n_outputs is None:
        eval_command(f"n = nargout(@{name_or_path});")
        n_outputs = int(jnp.squeeze(get_variable("n")))
    input_names = [f"i_{i}" for i in range(len(inputs))]
    output_names = [f"o_{i}" for i in range(n_outputs)]
    for (input_name, input) in zip(input_names, inputs):
        set_variable(input_name, input)
    eval_command(f"[{",".join(output_names)}]={name_or_path}({",".join(input_names)});")
    outputs = []
    for output_name in output_names:
        outputs.append(get_variable(output_name))
    eval_command("clear;")
    return outputs


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
    input_names: tuple[str, ...],
    command: str,
    output_names: tuple[str, ...],
    abstract_outputs: tuple[ShapeDtypeStruct, ...],
    use_file=bool,
) -> list[Array]:
    return list(jax.ffi.ffi_call(
        "run_matlab",
        abstract_outputs,
        input_layouts=[tuple(reversed(range(input.ndim))) for input in inputs],
        output_layouts=[tuple(reversed(range(abstract_output.ndim))) for abstract_output in abstract_outputs]
    )(
        *inputs,
        command_as_bytes=encode_string(command),
        input_names_as_bytes=encode_string_list(input_names),
        output_names_as_bytes=encode_string_list(output_names),
        use_file=use_file,
    ))


def run_matlab_abstract_eval(
    *inputs: ShapedArray,
    input_names: tuple[str, ...],
    command: str,
    output_names: tuple[str, ...],
    abstract_outputs: tuple[ShapeDtypeStruct, ...],
    use_file: bool,
) -> list[ShapedArray]:
    return [ShapedArray(shape=abstract_output.shape, dtype=abstract_output.dtype) for abstract_output in abstract_outputs]


def run_matlab_pp_rule(eqn: jax._src.core.JaxprEqn, context: jax._src.core.JaxprPpContext, settings: jax._src.core.JaxprPpSettings) -> jax._src.pretty_printer.Doc:
    return jax._src.core._pp_eqn(eqn, context, settings, ["input_names", "command", "output_names"])


def run_matlab(
    *,
    inputs: Sequence[Array],
    input_names: Sequence[str],
    command: str,
    output_names: Sequence[str],
    abstract_outputs: Sequence[ShapeDtypeStruct],
    use_file: bool = True,
) -> list[Array]:
    return list(run_matlab_p.bind(
        *inputs,
        input_names=tuple(input_names),
        command=command,
        output_names=tuple(output_names),
        abstract_outputs=tuple(abstract_outputs),
        use_file=use_file,
    ))


def run_and_time_matlab(
    *,
    inputs: Sequence[Array],
    input_names: Sequence[str],
    command: str,
    output_names: Sequence[str],
    abstract_outputs: Sequence[ShapeDtypeStruct],
    use_file: bool = True,
) -> list[Array]:
    start = time.time()

    runtime_name = "G7B2KX9M4J1Z5T8W3Q6Y"
    command = f"{runtime_name} = tic;\n{command}\n{runtime_name} = toc({runtime_name});"
    output_names = [runtime_name, *output_names]
    abstract_outputs = [ShapeDtypeStruct(shape=(1, 1), dtype=jnp.float64), *abstract_outputs]
    [runtime, *outputs] = run_matlab(
        inputs=inputs,
        input_names=input_names,
        command=command,
        output_names=output_names,
        abstract_outputs=abstract_outputs,
        use_file=use_file,
    )
    runtime = float(jnp.squeeze(runtime))
    jax.block_until_ready(outputs)

    overhead = time.time() - start - runtime

    print(f"runtime:  {runtime:.2f}")
    print(f"overhead: {overhead:.2f}")

    return outputs


run_matlab_p = Primitive("run_matlab")
run_matlab_p.multiple_results = True
run_matlab_p.def_impl(run_matlab_impl)
run_matlab_p.def_abstract_eval(run_matlab_abstract_eval)
jax._src.core.pp_eqn_rules[run_matlab_p] = run_matlab_pp_rule
jax.interpreters.mlir.register_lowering(run_matlab_p, jax.interpreters.mlir.lower_fun(run_matlab_impl, multiple_results=True))
