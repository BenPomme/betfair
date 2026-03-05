#!/usr/bin/env python3
"""
Train and evaluate a predictive sports model from CSV.

CSV schema (header required):
timestamp,base_prob,odds,label,spread_mean,imbalance,depth_total_eur,price_velocity,short_volatility,time_to_start_sec,in_play
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

from strategy.predictive_model import (
    PredictionExample,
    ResidualLogitModel,
    evaluate_predictions,
    walk_forward_backtest,
)


DEFAULT_FEATURES = [
    "spread_mean",
    "imbalance",
    "depth_total_eur",
    "price_velocity",
    "short_volatility",
    "time_to_start_sec",
    "in_play",
]


def load_examples(path: str, feature_names: List[str]) -> List[PredictionExample]:
    rows: List[PredictionExample] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            features = {name: float(r.get(name, 0.0)) for name in feature_names}
            rows.append(
                PredictionExample(
                    timestamp=str(r["timestamp"]),
                    base_prob=float(r["base_prob"]),
                    odds=float(r["odds"]),
                    label=int(r["label"]),
                    features=features,
                )
            )
    rows.sort(key=lambda x: x.timestamp)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prediction/training.csv")
    parser.add_argument("--output", default="models/predictive_model_v1.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    examples = load_examples(args.input, DEFAULT_FEATURES)
    if len(examples) < 20:
        print("Not enough rows in dataset. Need at least 20.")
        return 1

    split_idx = int(len(examples) * args.train_ratio)
    split_idx = min(max(split_idx, 1), len(examples) - 1)
    train = examples[:split_idx]
    test = examples[split_idx:]

    model = ResidualLogitModel(DEFAULT_FEATURES)
    model.fit(train, epochs=args.epochs, lr=args.lr)
    model.save(args.output)

    probs = [model.predict_proba(ex.base_prob, ex.features) for ex in test]
    labels = [ex.label for ex in test]
    odds = [ex.odds for ex in test]
    holdout = evaluate_predictions(
        probs=probs,
        labels=labels,
        odds=odds,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
    )

    wf, _ = walk_forward_backtest(
        examples=examples,
        train_window=max(50, min(400, len(examples) // 2)),
        test_window=max(20, min(100, len(examples) // 4)),
        epochs=args.epochs,
        lr=args.lr,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
    )

    print(f"Saved model: {Path(args.output).resolve()}")
    print(
        "Holdout metrics: "
        f"brier={holdout.brier:.4f} logloss={holdout.logloss:.4f} "
        f"acc={holdout.accuracy:.4f} bets={holdout.bets} "
        f"roi={holdout.roi:.4f} pnl_units={holdout.pnl_units:.2f}"
    )
    print(
        "Walk-forward metrics: "
        f"brier={wf.brier:.4f} logloss={wf.logloss:.4f} "
        f"acc={wf.accuracy:.4f} bets={wf.bets} "
        f"roi={wf.roi:.4f} pnl_units={wf.pnl_units:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
