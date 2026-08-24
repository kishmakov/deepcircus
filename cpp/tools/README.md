# cpp/tools

Generator-independent input sampling and bit utilities in namespace `tools`.
The library also provides the `expand_inputs` command-line interface used by
Python code.

This layer must not depend on `gen`, so lightweight callers can use it without
linking the generator.
