cmake_minimum_required(VERSION 3.16)

project(experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(experiment
    main.cpp
    scheme.h
    scheme.cpp
    reconstruct.h
    reconstruct.cpp
    # The exact decision-tree solvers, compiled straight into the experiment:
    # `cpp/` is a separate CMake project and `tools` depends on nothing else.
    ../cpp/tools/solver.cpp
)
target_include_directories(experiment PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools")

# `main.cpp` checks the operation table and the reconstruction it prints, so its
# asserts have to outlive the Release `-DNDEBUG`. Source properties land after
# the build-type flags, so the `-U` wins.
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-UNDEBUG")
