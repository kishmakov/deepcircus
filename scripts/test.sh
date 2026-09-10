#!/usr/bin/env bash
set -euo pipefail

# Builds the whole C++ tree with ASan/UBSan and every assert alive -- Debug, so
# no `NDEBUG` anywhere -- then runs the test suite and, after it, each
# executable once on a small input of its own. The executables' error handling
# is their asserts, and the suite does not reach those; a run does. Extra
# arguments go to GoogleTest.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/cpp/build-asan"

cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
    -G Ninja \
    -DDEEPCIRCUS_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build "$BUILD_DIR"

export ASAN_OPTIONS="detect_leaks=1:${ASAN_OPTIONS:-}"

"$BUILD_DIR/test/deepcircus_tests" "$@"

# Everything the executables write goes here and leaves with the run.
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo
echo "== data_generator: tt and general entries at bitness 8 and 13"
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m1 8 239 2 2 2
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m2 8 239 2 2 2
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m1 13 239 1 1 1
ls "$WORK_DIR"

echo
echo "== validation: reconstruct a random 10-input scheme"
"$BUILD_DIR/validation/validation" 10 122 | tail -2

echo
echo "== offline_server: serve the files above, driven by the Python client"
cd "$ROOT"
OFFLINE_SERVER="$BUILD_DIR/offline_server" PYTHONPATH="$ROOT" \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/test.py" "$WORK_DIR"

echo
echo "all executables ran with assertions enabled"
