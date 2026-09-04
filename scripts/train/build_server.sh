#!/usr/bin/env bash
set -euo pipefail

# Builds the daemon that serves `data/` to the training loop, then publishes it
# under `execs/` for `train_model.sh`.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="cpp/build/offline_server"
LINK="$ROOT/execs/offline_server"

# Same flags as `scripts/build.sh`, which drives the same tree; differing ones
# would reconfigure it on every alternation.
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" \
    -G Ninja \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"
cmake --build "$ROOT/cpp/build" --config Release --target offline_server

# Relative, so the link survives the repository moving; `-n` keeps a re-run from
# dropping the new link inside the old one.
mkdir -p "$ROOT/execs"
ln -sfn "../$BINARY" "$LINK"
echo "execs/offline_server -> $BINARY"
