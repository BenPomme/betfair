#!/usr/bin/env python3
"""
Backtest a trained predictive model against a CSV dataset.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is importable when script is run directly.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from strategy.predictive_model import PredictionExample, ResidualLogitModel, evaluate_predictions


FEATURES = [
    "spread_mean",
    "imbalance",
    "depth_total_eur",
    "price_velocity",
    "short_volatility",
    "time_to_start_sec",
    "in_play",
]


def load_examples(path: str) -> List[PredictionExample]:
    out: List[PredictionExample] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(
                PredictionExample(
                    timestamp=str(r["timestamp"]),
                    base_prob=float(r["base_prob"]),
                    odds=float(r["odds"]),
                    label=int(r["label"]),
                    features={name: float(r.get(name, 0.0)) for name in FEATURES},
                )
            )
    out.sort(key=lambda x: x.timestamp)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/predictive_model_v1.json")
    parser.add_argument("--input", default="data/prediction/training.csv")
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--stake", type=float, default=1.0)
    args = parser.parse_args()

    model = ResidualLogitModel.load(args.model)
    examples = load_examples(args.input)
    if not examples:
        print("Dataset is empty.")
        return 1

    probs = [model.predict_proba(ex.base_prob, ex.features) for ex in examples]
    labels = [ex.label for ex in examples]
    odds = [ex.odds for ex in examples]

    metrics = evaluate_predictions(
        probs=probs,
        labels=labels,
        odds=odds,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
    )
    print(
        f"Backtest metrics: brier={metrics.brier:.4f} "
        f"logloss={metrics.logloss:.4f} acc={metrics.accuracy:.4f} "
        f"bets={metrics.bets} roi={metrics.roi:.4f} pnl_units={metrics.pnl_units:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
