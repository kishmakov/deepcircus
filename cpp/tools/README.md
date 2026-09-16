# cpp/tools

Input sampling, shared M1 restriction graph storage, and bit utilities in
namespace `tools`, shared by the offline server and tests. Their interface is
[`sample.h`](sample.h).

[`cli.h`](cli.h) declares shared unsigned argument parsing for the generator
and visualizer; [`cli.cpp`](cli.cpp) implements it with assertions enabled in
Release. `Parse16`, `Parse32`, and `Parse64` accept optional inclusive bounds,
defaulting to the full range of their unsigned return type.
