#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmake -S . -B build \
    -G Ninja \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"
cmake --build build --config Release
