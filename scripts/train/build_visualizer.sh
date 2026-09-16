#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="cpp/build/train/graph_vis"
LINK="$ROOT/execs/graph_vis"

"$ROOT/scripts/build.sh" graph_vis

mkdir -p "$ROOT/execs"
ln -sfn "../$BINARY" "$LINK"
echo "execs/graph_vis -> $BINARY"
