#!/usr/bin/env bash
set -euo pipefail

# Builds just the `validation` target out of the shared `cpp/build` tree;
# `scripts/build.sh` builds it along with everything else.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" \
    -G Ninja \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"
cmake --build "$ROOT/cpp/build" --config Release --target validation
