#!/usr/bin/env bash
set -euo pipefail

# Plots what training left in the work directory of `conf/train.yaml`: for every
# `<model>_<bitness>.metrics.json`, that run's train and validation error against
# each other over its epochs, as one `.png` beside the JSON stacking three
# panels -- both scores together, then the depth score and the size score.
#
#     scripts/docs/plot_metrics.sh            # error on a linear Y axis
#     scripts/docs/plot_metrics.sh scale=log  # error on a logarithmic one

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SCALE="lin"
for argument in "$@"; do
    case "$argument" in
        scale=lin | scale=log) SCALE="${argument#scale=}" ;;
        *)
            echo "usage: $0 [scale=lin|scale=log]" >&2
            exit 1
            ;;
    esac
done

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/docs/plot_metrics.py" --scale "$SCALE"
