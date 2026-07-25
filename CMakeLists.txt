cmake_minimum_required(VERSION 3.18)
project(matlab4jax_cpp LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Matlab REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "import jax; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

nanobind_add_module(matlab4jax_cpp src/matlab4jax_cpp.cpp)

target_include_directories(matlab4jax_cpp PRIVATE
    ${JAX_FFI_INCLUDE_DIR}
    ${Matlab_INCLUDE_DIRS}
)

target_link_directories(matlab4jax_cpp PRIVATE
    ${Matlab_ROOT_DIR}/extern/bin/glnxa64
    ${Matlab_ROOT_DIR}/bin/glnxa64
)

target_link_libraries(matlab4jax_cpp PRIVATE
    MatlabEngine
    MatlabDataArray
)

install(TARGETS matlab4jax_cpp LIBRARY DESTINATION .)
