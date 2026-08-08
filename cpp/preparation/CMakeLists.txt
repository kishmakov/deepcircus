add_executable(offline_train_data_generator
    main.cpp
)

# Error handling here is asserts only, so they have to outlive the Release
# `-DNDEBUG`; source properties land after the build-type flags, so `-U` wins.
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-UNDEBUG")
