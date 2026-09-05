#!/usr/bin/env bash
set -euo pipefail

# Builds the daemon that serves `data/` to the training loop, then publishes it
# under `execs/` for `train_model.sh`.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="cpp/build/offline_server"
LINK="$ROOT/execs/offline_server"

"$ROOT/scripts/build.sh" offline_server

# Relative, so the link survives the repository moving; `-n` keeps a re-run from
# dropping the new link inside the old one.
mkdir -p "$ROOT/execs"
ln -sfn "../$BINARY" "$LINK"
echo "execs/offline_server -> $BINARY"
