#!/usr/bin/env bash
set -euo pipefail

exec perf report -i /tmp/circus/perf.data "$@"
