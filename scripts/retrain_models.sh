#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[retrain] training scoring model..."
.venv/bin/python -m strategy.train_scoring_model \
  --input-dir data/candidates \
  --output data/models/scoring_linear_v2.json || true

echo "[retrain] training fill model..."
.venv/bin/python -m strategy.fill_model \
  --input-dir data/candidates \
  --output data/models/fill_model_v1.json || true

echo "[retrain] validating performance gates..."
.venv/bin/python scripts/validate_performance_gates.py \
  --input-dir data/candidates \
  --output-dir data/reports/performance_gates \
  --min-samples 200 || true

echo "[retrain] retraining prediction models..."
.venv/bin/python scripts/retrain_prediction_models.py || true

echo "[retrain] done"
