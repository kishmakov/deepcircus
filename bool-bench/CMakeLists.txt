cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")

project(deepcircus_bool_bench LANGUAGES C CXX)

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

add_library(bb
    src/aig.cpp
    src/aig.h
    src/decision_tree.cpp
    src/decision_tree.h
    src/dataset.cpp
    src/bool_bench.cpp
    src/bool_bench.h
    src/table.h
    src/table.cpp
    src/tree.cpp
    src/tree.h
    src/utils.cpp
    src/utils.h
)

target_include_directories(bb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(bb PRIVATE aiger Threads::Threads)
