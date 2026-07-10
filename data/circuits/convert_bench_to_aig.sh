#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 file.bench" >&2
  exit 2
fi

f="$1"
abc -q "read_bench $f; strash; write_aiger ${f%.bench}.aig"
