# cpp/common

Generator-independent code shared by the C++ executables.

| Directory | What is in it |
| --- | --- |
| [`tools/`](tools/README.md) | exact solvers and shared randomness |
| [`func/`](func/README.md) | truth-table cases and generation helpers |
| [`offline/`](offline/README.md) | offline training-file I/O |

Most of this directory forms the `common` library. Truth-table generation that
depends on generator types is built into `gen` instead.
