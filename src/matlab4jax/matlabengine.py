from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

engine = None


def get_engine() -> Any:
    import matlab.engine
    global engine
    if engine is None:
        engine = matlab.engine.start_matlab()
    return engine


def eval_command(command: str, /) ->None:
    engine = get_engine()
    engine.eval(command, nargout=0)


def set_variable(name: str, value: ArrayLike, /) ->None:
    engine = get_engine()
    engine.workspace[name] = np.asarray(value)


def get_variable(name: str, /) ->Array:
    engine = get_engine()
    return jnp.asarray(engine.workspace[name])


def eval_function(name_or_path: str | Path, inputs: Sequence[ArrayLike], /, *, n_outputs: int | None = None) -> list[Array]:
    engine = get_engine()
    engine.eval("clear;", nargout=0)
    if isinstance(name_or_path, Path) or name_or_path.endswith(".m"):
        name_or_path = Path(name_or_path)
        assert name_or_path.exists()
        engine.eval(f"addpath('{name_or_path.parent}');", nargout=0)
        name_or_path = name_or_path.stem
    inputs = [np.asarray(input) for input in inputs]
    if n_outputs is None:
        n_outputs = int(engine.feval(f"nargout", name_or_path, nargout=1))
    outputs = engine.feval(name_or_path, *inputs, nargout=n_outputs)
    if n_outputs == 1:
        outputs = [outputs]
    outputs = [jnp.asarray(output) for output in outputs]
    engine.eval("clear;", nargout=0)
    return outputs
