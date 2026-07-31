cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")

project(deepcircus_generator LANGUAGES C CXX)

find_package(Threads REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    aiger
    GIT_REPOSITORY https://github.com/arminbiere/aiger.git
    GIT_TAG rel-1.9.20
)
FetchContent_MakeAvailable(aiger)

add_library(aiger STATIC "${aiger_SOURCE_DIR}/aiger.c")
target_include_directories(aiger PUBLIC "${aiger_SOURCE_DIR}")
# aiger's REALLOCN grows fresh arrays with memcpy(res, NULL, 0), which trips
# UBSan's nonnull-attribute check; silence it for this vendored file only.
target_compile_options(aiger PRIVATE -fno-sanitize=nonnull-attribute)

# Standalone bit/sampling/solving pieces. Deliberately does not link `gen`:
# tools knows nothing about cases, tables, trees, or circuits, so the include
# direction runs generator -> tools only.
add_library(tools
    tools/sample.cpp
    tools/sample.h
    tools/solver.cpp
    tools/solver.h
)
target_include_directories(tools PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tools")

add_library(gen
    generator/aig.cpp
    generator/aig.h
    generator/case.cpp
    generator/case.h
    generator/generator.cpp
    generator/generator.h
    generator/table.h
    generator/table.cpp
    generator/tree.cpp
    generator/tree.h
    generator/utils.cpp
    generator/utils.h
)

target_include_directories(gen PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/generator")
target_link_libraries(gen PUBLIC tools)
target_link_libraries(gen PRIVATE aiger)

add_library(producer
    producer/task_queue.cpp
    producer/task_queue.h
    producer/thread_pool.cpp
    producer/thread_pool.h
)
target_include_directories(producer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/producer")
target_link_libraries(producer PUBLIC gen Threads::Threads)

add_executable(generator_server
    server/daemon.cpp
    server/daemon.h
    server/server.cpp
)
target_include_directories(generator_server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/server")
target_link_libraries(generator_server PRIVATE producer Threads::Threads)

add_executable(expand_inputs tools/expand_inputs.cpp)
target_link_libraries(expand_inputs PRIVATE tools)

option(DEEPCIRCUS_BUILD_TESTS "Build the Google Test suite" ON)
if(DEEPCIRCUS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
