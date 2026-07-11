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

add_library(gen
    aig.cpp
    aig.h
    decision_tree.cpp
    decision_tree.h
    dataset.cpp
    generator.cpp
    generator.h
    table.h
    table.cpp
    tree.cpp
    tree.h
    utils.cpp
    utils.h
)

target_include_directories(gen PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(gen PRIVATE aiger Threads::Threads)

add_executable(generator_server server.cpp)
target_link_libraries(generator_server PRIVATE gen Threads::Threads)
