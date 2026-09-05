#!/usr/bin/env bash
set -euo pipefail

# Trains one model at one bitness on the offline data prepared for it:
#
#     scripts/train/train_model.sh m1 8
#
# Everything else about the run is `conf/train.yaml`. The weights and the
# per-epoch metrics land in that config's `work_dir`; `data/` is only read.
# Above bitness 12, M2 needs M2 at n-1; M1 needs M1 at n-1 and M2 at n.

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <m1|m2> <bitness>" >&2
    exit 1
fi

MODEL="$1"
BITNESS="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SERVER="$ROOT/execs/offline_server"

if [ "$MODEL" != "m1" ] && [ "$MODEL" != "m2" ]; then
    echo "error: model must be m1 or m2, got $MODEL" >&2
    exit 1
fi

# Resolves the symlink, so a stale one is caught here rather than mid-run.
if [ ! -x "$SERVER" ]; then
    echo "error: ${SERVER#"$ROOT"/} is not built; run scripts/train/build_server.sh first" >&2
    exit 1
fi

TAG="$(printf '%s_%02d' "$MODEL" "$BITNESS")"
for suffix in train val; do
    if [ ! -f "$ROOT/data/$TAG.$suffix" ]; then
        echo "error: data/$TAG.$suffix is missing; run scripts/prep/generate_train_data.sh first" >&2
        exit 1
    fi
done

cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m src.train --model "$MODEL" --bitness "$BITNESS"
