add_executable(validation
    main.cpp
    scheme.h
    scheme.cpp
    reconstruct.h
    reconstruct.cpp
)

# The exact decision-tree solvers come from `tools`, which knows nothing about
# cases, tables, trees, or circuits; validation stays on that side of the line
# and does not link `gen`.
target_link_libraries(validation PRIVATE tools)

# `main.cpp` checks the operation table and the reconstruction it prints, so its
# asserts have to outlive the Release `-DNDEBUG`. Source properties land after
# the build-type flags, so the `-U` wins.
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-UNDEBUG")
