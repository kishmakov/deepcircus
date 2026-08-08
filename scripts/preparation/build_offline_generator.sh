#!/usr/bin/env bash
set -euo pipefail

# Builds the one target `scripts/build.sh` also covers, then publishes it under
# `execs/` -- the name `prepare_offline_train_data.sh` runs.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="cpp/build/preparation/offline_train_data_generator"
LINK="$ROOT/execs/offline_train_data_generator"

# Same flags as `scripts/build.sh`, which drives the same tree; differing ones
# would reconfigure it on every alternation.
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" \
    -G Ninja \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"
cmake --build "$ROOT/cpp/build" --config Release --target offline_train_data_generator

# Relative, so the link survives the repository moving; `-n` keeps a re-run from
# dropping the new link inside the old one.
mkdir -p "$ROOT/execs"
ln -sfn "../$BINARY" "$LINK"
echo "execs/offline_train_data_generator -> $BINARY"
