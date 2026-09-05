FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(deepcircus_tests
    test_binary_tree.cpp
    test_offline_data.cpp
    test_preparation.cpp
    test_random.cpp
    test_sample.cpp
    test_serving.cpp
    test_solver.cpp
    test_table.cpp
    test_tree.cpp
    ../prep/sampler.cpp
)
target_include_directories(deepcircus_tests PRIVATE ../prep)
target_link_libraries(deepcircus_tests PRIVATE common tools serving GTest::gtest_main)

gtest_discover_tests(deepcircus_tests)
