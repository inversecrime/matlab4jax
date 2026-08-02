import os
from pathlib import Path
import subprocess


EXPECTED_MATLAB_RELEASE = "R2026a"


def get_matlab_release():
    matlab_root = os.environ.get("MATLAB_ROOT")

    if matlab_root is None:
        matlab = subprocess.run(
            ["which", "matlab"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        if not matlab:
            return None

        matlab_root = Path(matlab).resolve().parents[1]

    else:
        matlab_root = Path(matlab_root)

    return matlab_root.name


def check_matlab_version():
    matlab_release = get_matlab_release()

    if matlab_release != EXPECTED_MATLAB_RELEASE:
        raise ImportError(
            f"This matlab4jax wheel was built for MATLAB "
            f"{EXPECTED_MATLAB_RELEASE}, but found {matlab_release}. "
            "Install matlab4jax from source for another MATLAB release."
        )