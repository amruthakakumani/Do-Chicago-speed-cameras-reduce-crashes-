#!/usr/bin/env bash
# End-to-end rebuild: download -> clean -> estimate -> verify -> test.
#
# Usage:
#   bash scripts/run_all.sh            # full run
#   QUICK=1 bash scripts/run_all.sh    # development run (fewer bootstrap draws)
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "== 1/4: download Chicago Data Portal datasets =="
"$PYTHON" scripts/download_data.py

echo "== 2/4: run the analysis pipeline =="
if [[ "${QUICK:-0}" == "1" ]]; then
  "$PYTHON" -m chicago_cameras.pipeline --config config/analysis.yaml --quick
else
  "$PYTHON" -m chicago_cameras.pipeline --config config/analysis.yaml
fi

echo "== 3/4: verify the original study's published estimates =="
"$PYTHON" scripts/verify_replication.py

if [[ "${RUN_ML:-0}" == "1" ]]; then
  echo "== 4/5: train and evaluate the LightGBM forecaster =="
  "$PYTHON" scripts/train_lightgbm.py --trials "${ML_TRIALS:-40}" --timeout "${ML_TIMEOUT:-900}"
  "$PYTHON" scripts/evaluate_lightgbm.py
  "$PYTHON" scripts/build_prediction_report.py
  echo "== 5/5: run the test suite =="
else
  echo "== 4/4: run the test suite =="
fi

if [[ "${RUN_ML_V2:-0}" == "1" ]]; then
  echo "== V2: train, tune, intervals, geographic holdout, recursive benchmark =="
  "$PYTHON" scripts/train_lightgbm_v2.py --trials "${ML_TRIALS:-40}" --timeout "${ML_TIMEOUT:-900}"
  "$PYTHON" scripts/evaluate_recursive_v2.py
fi

"$PYTHON" -m pytest -q

echo
echo "Done. Results: results/  |  Figures: results/figures/  |  Report: docs/REPORT.md"
