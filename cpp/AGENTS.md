# C++

The architecture is documented per directory, starting at
[`cpp/README.md`](README.md), and the project-wide boundaries are in the root
`AGENTS.md`. What follows is what the C++ side needs on top of them.

- Release builds define `NDEBUG`, so assume an assert vanishes there: never put
  a side effect inside one.
- Where an assert *is* the error handling rather than debug scaffolding,
  undefine `NDEBUG` for that file in its CMake entry (source properties land
  after the build-type flags, so the `-U` wins) instead of turning the check
  into an `if`. Four files do this today -- `common/offline/read_write.cpp`,
  `preparation/main.cpp`, `preparation/sampler.cpp`, and `validation/main.cpp`.
- A new source file goes in the target its dependencies allow, not the one that
  is convenient: `common` and `tools` may not grow an edge back to `gen`.
- Formatting and linting come from `.clang-format` and `.clang-tidy` here.
