from pathlib import Path

from setuptools import setup, Extension, find_packages


MATLAB_ROOT = str(sorted(Path("/usr/local/MATLAB").iterdir())[-1])

ext_modules = [
    Extension(
        name="matlab4jax.matlab4jax_cpp",
        sources=[
            "cpp/main.cpp",
        ],
        include_dirs=[
            "lib/pybind11/include",
            "lib/xla",
            f"{MATLAB_ROOT}/extern/include",
        ],
        library_dirs=[
            f"{MATLAB_ROOT}/extern/bin/glnxa64",
        ],
        libraries=[
            "MatlabDataArray",
            "MatlabEngine",
        ],
        extra_compile_args=[
            "-O3",
            "-std=c++17",
        ],
        extra_link_args=[
            f"-Wl,-rpath,{MATLAB_ROOT}/extern/bin/glnxa64",
            "-pthread",
        ],
    )
]

setup(
    name="matlab4jax",
    version="0.0.1",
    author="inversecrime",
    url="https://github.com/inversecrime/matlab4jax",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    python_requires=">=3.12",
    zip_safe=False,
)
