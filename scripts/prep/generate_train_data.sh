#!/usr/bin/env bash
set -euo pipefail

# Makes sure the offline training data is in `data/`, generating what is
# missing. Safe to re-run.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$ROOT/conf/preparation.yaml"
DATA_DIR="$ROOT/data"
# Published by `build_offline_generator.sh`.
GENERATOR="$ROOT/execs/data_generator"

SERIES=(1 2)
# `rand` draws both truth tables at random, `small` builds each entry around a
# witness tree so its exact target lands in the configured small-size range.
SOURCES=(rand small)

# The dotted paths this script needs, in the order the reader prints them.
CONFIG_KEYS=(
    work_dir
    seed
    entries
    full_tables.bitness.from
    full_tables.bitness.to
    full_tables.small_sizes.from
    full_tables.small_sizes.to
)

# Resolves every path in one interpreter start, so the config's hierarchy and
# its comments stay OmegaConf's business instead of something bash parses. A
# missing path is fatal rather than defaulted: generating under a silently wrong
# seed would look like success.
read_config() {
    "$ROOT/.venv/bin/python" - "$CONFIG" "${CONFIG_KEYS[@]}" <<'PY'
import sys

from omegaconf import OmegaConf

path, keys = sys.argv[1], sys.argv[2:]
config = OmegaConf.load(path)
for key in keys:
    value = OmegaConf.select(config, key, default=None)
    assert value is not None, f"no '{key}' in {path}"
    print(value)
PY
}

# Capturing into a variable is what lets `set -e` see a failed read; splitting
# straight into `readarray` would swallow the exit status and leave the values
# below half-assigned.
CONFIG_VALUES="$(read_config)"
readarray -t values <<< "$CONFIG_VALUES"
if [ "${#values[@]}" -ne "${#CONFIG_KEYS[@]}" ]; then
    echo "error: read ${#values[@]} of ${#CONFIG_KEYS[@]} values from ${CONFIG#"$ROOT"/}" >&2
    exit 1
fi

declare -A config
for index in "${!CONFIG_KEYS[@]}"; do
    config["${CONFIG_KEYS[$index]}"]="${values[$index]}"
done

# The generator owns this directory; `data/` only gains files that came out of a
# finished run.
STAGE_DIR="${config[work_dir]}"
SEED="${config[seed]}"
ENTRIES="${config[entries]}"
BITNESS_FROM="${config[full_tables.bitness.from]}"
BITNESS_TO="${config[full_tables.bitness.to]}"
SMALL_SIZE_FROM="${config[full_tables.small_sizes.from]}"
SMALL_SIZE_TO="${config[full_tables.small_sizes.to]}"

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
