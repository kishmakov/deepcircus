cmake_minimum_required(VERSION 3.16)

project(sizes_experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(sizes_experiment
    sizes.cpp
    scheme.h
    scheme.cpp
    # The exact decision-tree solvers, compiled straight into the experiment:
    # `cpp/` is a separate CMake project and `tools` depends on nothing else.
    ../cpp/tools/solver.cpp
)
target_include_directories(sizes_experiment PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools")
