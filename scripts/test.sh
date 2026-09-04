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
echo "== data_generator: a few solved entries of each model at bitness 8"
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m1 8 239 4 0 2
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m2 8 239 4 0 2
ls "$WORK_DIR"

echo
echo "== expand_inputs: two 4-bit base sequences into 2 x 4 points"
EXPANDED="$(printf '0101\n1100\n' | "$BUILD_DIR/expand_inputs" 2 4)"
[ "${#EXPANDED}" -eq 32 ] || { echo "error: expected 32 bits, got ${#EXPANDED}" >&2; exit 1; }
echo "$EXPANDED"

echo
echo "== validation: reconstruct a random 10-input scheme"
"$BUILD_DIR/validation/validation" 10 122 | tail -2

echo
echo "== offline_server: serve the files above, driven by the Python client"
cd "$ROOT"
OFFLINE_SERVER="$BUILD_DIR/offline_server" "$ROOT/.venv/bin/python" - "$WORK_DIR" <<'PY'
import sys
from pathlib import Path

from src.daemon.client import VALIDATION_EPOCH, Client

client = Client("m1", 8, Path(sys.argv[1]), 239, 2, 4)
assert client.sizes.train_cases == 4, client.sizes
assert client.sizes.validation_cases == 2, client.sizes
validation = client.fetch(VALIDATION_EPOCH)
epoch = client.fetch(1)
assert validation.values.shape == (2, 26), validation.values.shape
assert epoch.values.shape == (4, 26), epoch.values.shape
assert epoch.targets.shape == (4, 2), epoch.targets.shape
# Closing hangs up, and the daemon's exit status -- LeakSanitizer included --
# is asserted on the way out.
client.close()
print("served 2 validation and 4 training cases, daemon exited cleanly")
PY

echo
echo "all executables ran with assertions enabled"
