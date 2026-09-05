#!/usr/bin/env bash
set -euo pipefail

# Configures the release C++ tree once and builds either all targets or the
# targets named on the command line.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/cpp/build"

cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
    -G Ninja \
    -DBUILD_SHARED_LIBS=ON \
    -DDEEPCIRCUS_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"

if [ "$#" -eq 0 ]; then
    cmake --build "$BUILD_DIR" --config Release
else
    cmake --build "$BUILD_DIR" --config Release --target "$@"
fi
