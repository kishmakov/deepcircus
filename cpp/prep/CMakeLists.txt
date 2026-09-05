add_executable(data_generator
    main.cpp
    sampler.cpp
)
target_link_libraries(data_generator PRIVATE common Threads::Threads)
target_include_directories(data_generator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Error handling here is asserts only -- and in `sampler.cpp` an assert is also
# what bounds the rejection loops -- so they have to outlive the Release
# `-DNDEBUG`; source properties land after the build-type flags, so `-U` wins.
set_source_files_properties(main.cpp sampler.cpp PROPERTIES COMPILE_OPTIONS "-UNDEBUG")
