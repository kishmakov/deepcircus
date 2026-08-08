#!/usr/bin/env bash
set -euo pipefail

# Makes sure the offline training data is in `data/`, generating what is
# missing. Safe to re-run.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$ROOT/conf/preparation.conf"
DATA_DIR="$ROOT/data"
# Published by `build_offline_generator.sh`.
GENERATOR="$ROOT/execs/offline_train_data_generator"

SERIES=(1 2)
# `rand` draws both truth tables at random, `small` builds each entry around a
# witness tree so its exact target lands in the configured small-size range.
SOURCES=(rand small)

# Reads one `key: value` line. A missing key is fatal rather than defaulted:
# generating under a silently wrong seed would look like success.
config_value() {
    local key="$1" name value
    while read -r name value; do
        if [ "$name" = "$key:" ]; then
            printf '%s\n' "$value"
            return 0
        fi
    done < "$CONFIG"
    echo "error: no '$key' in ${CONFIG#"$ROOT"/}" >&2
    return 1
}

# The generator owns this directory; `data/` only gains files that came out of a
# finished run.
STAGE_DIR="$(config_value work_dir)"
SEED="$(config_value seed)"
ENTRIES="$(config_value entries)"
BITNESS_FROM="$(config_value bitness_from)"
BITNESS_TO="$(config_value bitness_to)"
SMALL_SIZE_FROM="$(config_value small_size_from)"
SMALL_SIZE_TO="$(config_value small_size_to)"

series_files() {
    local bitness="$1"
    local series source
    for source in "${SOURCES[@]}"; do
        for series in "${SERIES[@]}"; do
            printf 's%s_%02d_%s.bin\n' "$series" "$bitness" "$source"
        done
    done
}

# One missing file condemns its whole bitness: the generator emits the series
# and the sources together, so there is no asking it for just one of them.
missing=()
for ((bitness = BITNESS_FROM; bitness <= BITNESS_TO; bitness++)); do
    readarray -t names < <(series_files "$bitness")
    for name in "${names[@]}"; do
        if [ ! -f "$DATA_DIR/$name" ]; then
            missing+=("$bitness")
            break
        fi
    done
done

if [ ${#missing[@]} -eq 0 ]; then
    echo "offline train data complete in ${DATA_DIR#"$ROOT"/}"
    exit 0
fi

echo "missing offline train data for bitness: ${missing[*]}"

# Resolves the symlink, so a stale one is caught here rather than at the call.
if [ ! -x "$GENERATOR" ]; then
    echo "error: ${GENERATOR#"$ROOT"/} is not built; run scripts/preparation/build_offline_generator.sh first" >&2
    exit 1
fi

mkdir -p "$DATA_DIR" "$STAGE_DIR"

for bitness in "${missing[@]}"; do
    echo "generating $ENTRIES entries for bitness $bitness into $STAGE_DIR (seed $SEED)"
    "$GENERATOR" "$STAGE_DIR" "$bitness" "$SEED" "$ENTRIES" "$SMALL_SIZE_FROM" "$SMALL_SIZE_TO"

    readarray -t names < <(series_files "$bitness")
    for name in "${names[@]}"; do
        if [ ! -f "$STAGE_DIR/$name" ]; then
            echo "error: ${GENERATOR#"$ROOT"/} did not produce $STAGE_DIR/$name" >&2
            exit 1
        fi
        mv "$STAGE_DIR/$name" "$DATA_DIR/$name"
        echo "  -> ${DATA_DIR#"$ROOT"/}/$name"
    done
done

echo "offline train data ready in ${DATA_DIR#"$ROOT"/}"
