# cpp

The compiled part of the project: data generation, serving, offline data
preparation, validation, and tests.

| Directory | What is in it |
| --- | --- |
| [`common/`](common/README.md) | generator-independent code shared by the executables |
| [`tools/`](tools/README.md) | input sampling and bit utilities |
| [`generator/`](generator/README.md) | synchronous generation of cases and tensors |
| [`producer/`](producer/README.md) | parallel production and scheduling of generated data |
| [`server/`](server/README.md) | the daemon that feeds Python its training data |
| [`preparation/`](preparation/README.md) | offline training-data generation |
| [`validation/`](validation/README.md) | reconstruction and validation |
| [`test/`](test/README.md) | the C++ test suite |

[`CMakeLists.txt`](CMakeLists.txt) defines the libraries and executables.
The scripts in [`../scripts/`](../scripts/README.md) are the usual entry points
for building and testing them.
