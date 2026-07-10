# Bool Bench Notes

- Keep this directory small and dependency-light; it is used as a C/C++ generator with a thin Python helper.
- Preserve the public C ABI in `bool_bench.h`: keep exported functions `extern "C"` compatible and avoid C++-only types there.
- Prefer straightforward implementations over new abstractions unless they remove real duplication.
- Keep common functionality (such as RNG preparation or random bit sampling) in `src/utils.{h,cpp}`.
- When changing behavior, update nearby C++ and Python entry points together if they expose the same generator concept.
- Check builds through the local `CMakeLists.txt` path when edits touch compiled code.
