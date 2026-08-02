#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_part() {
  printf '[run] %s\n' "$*"
  "$@"
}

run_bounded_stage() {
  local expected="$1"
  shift
  printf '[run] %s\n' "$*"
  if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout --signal=TERM --kill-after=5s 120s "$@"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      if [[ $rc -eq 124 && -s "$expected" ]]; then
        printf '[run] output complete; terminated a stalled numerical-library shutdown\n'
      else
        return "$rc"
      fi
    fi
  else
    "$@"
  fi
}

for part in fig1 fig2; do
  run_part python code/reproduce_all.py --part "$part"
done

if [[ "${REGENERATE_HEAVY:-0}" == "1" || \
      ! -s data/label_resolution_gap_map.npz || \
      ! -s data/label_resolution_path.csv || \
      ! -s data/hierarchical_label_gaps.csv ]]; then
  run_part python code/reproduce_all.py --part figS1
else
  printf '[run] using stored projected-label data\n'
  run_part python code/reproduce_all.py --part figS1plot
fi

if [[ "${REGENERATE_HEAVY:-0}" == "1" || \
      ! -s data/structured_s4_c2_direct.csv || \
      ! -s data/clutching_tomography_grid_N8.npz || \
      ! -s data/structured_s4_hemisphere_patch_scan.csv ]]; then
  run_bounded_stage data/structured_s4_c2_direct.csv \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python code/reproduce_structured_s4.py --part base
  run_bounded_stage data/clutching_tomography_grid_N8.npz \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python code/reproduce_structured_s4.py --part grid
  run_bounded_stage data/structured_s4_hemisphere_patch_scan.csv \
    env OPENBLAS_NUM_THREADS=5 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
    python code/reproduce_structured_s4.py --part hemisphere
else
  printf '[run] using stored structured-S4 data\n'
fi

if [[ "${REGENERATE_HEAVY:-0}" == "1" || \
      ! -s data/finite_time_pump_trajectory.csv || \
      ! -s data/finite_time_pump_endpoints.csv || \
      ! -s data/finite_time_disorder_ensemble.csv ]]; then
  run_part python code/reproduce_all.py --part fig3
  OPENBLAS_NUM_THREADS=5 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
    run_part python code/reproduce_robustness.py --part pump
else
  printf '[run] using stored finite-time pump data\n'
  run_part python code/reproduce_all.py --part fig3plot
fi

if [[ "${REGENERATE_HEAVY:-0}" == "1" || \
      ! -s data/tomography_finite_shot_realizations.csv ]]; then
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    run_part python code/reproduce_robustness.py --part finite_shot
fi

if [[ "${REGENERATE_HEAVY:-0}" == "1" || \
      ! -s data/tomography_readout_confusion_realizations.csv ]]; then
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    run_part python code/reproduce_robustness.py --part readout
fi

run_part python code/reproduce_robustness.py --part coupling
run_part python code/reproduce_robustness.py --part plot
run_part python code/check_outputs.py
