#!/usr/bin/env bash
set -euo pipefail

# Trains one model at one bitness on the offline data prepared for it:
#
#     scripts/train/train_model.sh m1 8
#
# Everything else about the run is `conf/train.yaml`. The run reads `data/` and
# writes the weights and the per-epoch metrics into that config's `work_dir`;
# once it is over, the best weights are copied into `data/` under the same name,
# so a wiped scratch directory does not cost a trained coordinate.
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
"$ROOT/.venv/bin/python" -m src.train --model "$MODEL" --bitness "$BITNESS"

# Where the run left its best weights, asked of the same config the run read.
BEST="$("$ROOT/.venv/bin/python" - "$MODEL" "$BITNESS" <<'PY'
import sys

from src.config import load_train_config

config = load_train_config(sys.argv[1], int(sys.argv[2]))
print(config.best_checkpoint_path())
PY
)"

if [ ! -f "$BEST" ]; then
    echo "error: the run left no $BEST" >&2
    exit 1
fi

# Kept under the name the run wrote, so restoring a scratch directory emptied
# between runs is the copy back: bootstrapping a higher bitness looks for
# exactly this name in `work_dir`. Copied through a temporary of its own,
# because `work_dir` and `data/` need not share a filesystem.
NAME="$(basename "$BEST")"
cp "$BEST" "$ROOT/data/.tmp-$NAME"
mv "$ROOT/data/.tmp-$NAME" "$ROOT/data/$NAME"
echo "kept: data/$NAME"
