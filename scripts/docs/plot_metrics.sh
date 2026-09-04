#!/usr/bin/env bash
set -euo pipefail

# Plots what training left in the work directory of `conf/train.yaml`: one PNG
# per `<model>_<bitness>.metrics.json`, holding that run's train and validation
# RMSE against each other over its epochs. Each plot is written beside the JSON
# it was read from.
#
#     scripts/docs/plot_metrics.sh            # RMSE on a linear Y axis
#     scripts/docs/plot_metrics.sh scale=log  # RMSE on a logarithmic one

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
