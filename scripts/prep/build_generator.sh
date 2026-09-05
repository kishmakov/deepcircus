#!/usr/bin/env bash
set -euo pipefail

# Builds the one target `scripts/build.sh` also covers, then publishes it under
# `execs/` for `generate_train_data.sh`.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="cpp/build/prep/data_generator"
LINK="$ROOT/execs/data_generator"

"$ROOT/scripts/build.sh" data_generator

# Relative, so the link survives the repository moving; `-n` keeps a re-run from
# dropping the new link inside the old one.
mkdir -p "$ROOT/execs"
ln -sfn "../$BINARY" "$LINK"
echo "execs/data_generator -> $BINARY"
