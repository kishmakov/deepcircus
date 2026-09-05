#!/usr/bin/env bash
set -euo pipefail

# Generates the M1 and M2 train/validation files requested by
# `conf/preparation.yaml`. Complete pairs are left untouched on re-runs.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$ROOT/conf/preparation.yaml"
DATA_DIR="$ROOT/data"
GENERATOR="$ROOT/execs/data_generator"

read_config() {
    "$ROOT/.venv/bin/python" - "$CONFIG" <<'PY'
import sys

from yaml import safe_load

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    config = safe_load(f)
assert isinstance(config, dict), path
assert isinstance(config.get("work_dir"), str) and "\n" not in config["work_dir"], "work_dir"
assert type(config.get("seed")) is int and 0 <= config["seed"] <= 0xFFFFFFFFFFFFFFFF, "seed"

print(config["work_dir"])
print(config["seed"])
for model in ("m1", "m2"):
    jobs = config.get(model)
    assert isinstance(jobs, dict), model
    for bitness_key, counts in sorted(jobs.items(), key=lambda item: int(item[0])):
        bitness = int(bitness_key)
        assert 8 <= bitness <= 255, (model, bitness)
        assert isinstance(counts, dict), (model, bitness)
        train = counts.get("train")
        validation = counts.get("val")
        assert isinstance(train, dict), (model, bitness, "train")
        assert isinstance(validation, dict), (model, bitness, "val")

        solved = train.get("solved")
        unsolved = train.get("unsolved")
        val_solved = validation.get("solved")
        for name, value in (("solved", solved), ("unsolved", unsolved), ("val.solved", val_solved)):
            assert type(value) is int and 0 <= value <= 0xFFFFFFFF, (model, bitness, name)
        assert solved + unsolved <= 0xFFFFFFFF, (model, bitness, "train")
        assert bitness > 12 or unsolved == 0, (model, bitness, "unsolved below bitness 13")
        print(f"{model}\t{bitness}\t{solved}\t{unsolved}\t{val_solved}")
PY
}

CONFIG_LINES="$(read_config)"
readarray -t lines <<< "$CONFIG_LINES"
if [ "${#lines[@]}" -lt 2 ]; then
    echo "error: failed to read ${CONFIG#"$ROOT"/}" >&2
    exit 1
fi

STAGE_DIR="${lines[0]}"
SEED="${lines[1]}"
jobs=("${lines[@]:2}")
missing=()

for job in "${jobs[@]}"; do
    IFS=$'\t' read -r model bitness train_solved train_unsolved val_solved <<< "$job"
    tag="$(printf '%02d' "$bitness")"
    train="$model"_"$tag".train
    validation="$model"_"$tag".val
    if [ ! -f "$DATA_DIR/$train" ] || [ ! -f "$DATA_DIR/$validation" ]; then
        missing+=("$job")
    fi
done

if [ "${#missing[@]}" -eq 0 ]; then
    echo "offline train and validation data complete in ${DATA_DIR#"$ROOT"/}"
    exit 0
fi

if [ ! -x "$GENERATOR" ]; then
    echo "error: ${GENERATOR#"$ROOT"/} is not built; run scripts/prep/build_generator.sh first" >&2
    exit 1
fi

mkdir -p "$DATA_DIR" "$STAGE_DIR"

for job in "${missing[@]}"; do
    IFS=$'\t' read -r model bitness train_solved train_unsolved val_solved <<< "$job"
    tag="$(printf '%02d' "$bitness")"
    echo "generating $model bitness $bitness: train $train_solved solved + $train_unsolved unsolved, val $val_solved"
    "$GENERATOR" "$STAGE_DIR" "$model" "$bitness" "$SEED" "$train_solved" "$train_unsolved" "$val_solved"

    for suffix in train val; do
        name="$model"_"$tag"."$suffix"
        if [ ! -f "$STAGE_DIR/$name" ]; then
            echo "error: ${GENERATOR#"$ROOT"/} did not produce $STAGE_DIR/$name" >&2
            exit 1
        fi
        mv "$STAGE_DIR/$name" "$DATA_DIR/$name"
        echo "  -> ${DATA_DIR#"$ROOT"/}/$name"
    done
done

echo "offline train and validation data ready in ${DATA_DIR#"$ROOT"/}"
