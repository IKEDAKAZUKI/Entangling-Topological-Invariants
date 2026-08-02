#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ "${REGENERATE_ALL:-0}" == "1" ]]; then
  code/clean_generated.sh
  export REGENERATE_HEAVY=1
fi

code/run_calculations.sh
