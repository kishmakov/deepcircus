FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(deepcircus_tests
    test_bit_matrix.cpp
    test_case.cpp
    test_circuits.cpp
    test_table.cpp
    test_thread_pool.cpp
    test_task_queue.cpp
    test_tree.cpp
    test_utils.cpp
)
target_link_libraries(deepcircus_tests PRIVATE producer GTest::gtest_main)

gtest_discover_tests(deepcircus_tests)
