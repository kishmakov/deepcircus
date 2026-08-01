#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"

cmake -S "$EXPERIMENT_DIR" -B "$EXPERIMENT_DIR/build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"
cmake --build "$EXPERIMENT_DIR/build" --config Release --target experiment
