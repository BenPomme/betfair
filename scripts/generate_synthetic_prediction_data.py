#!/usr/bin/env python3
"""
Generate a synthetic labeled dataset for validating predictive model tooling.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def clip01(x: float) -> float:
    return max(1e-4, min(1 - 1e-4, x))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/prediction/training.csv")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp",
        "base_prob",
        "odds",
        "label",
        "spread_mean",
        "imbalance",
        "depth_total_eur",
        "price_velocity",
        "short_volatility",
        "time_to_start_sec",
        "in_play",
    ]

    start = datetime.now(timezone.utc) - timedelta(days=180)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(args.rows):
            ts = start + timedelta(minutes=10 * i)
            spread_mean = max(0.0, random.gauss(0.04, 0.02))
            imbalance = random.uniform(-0.8, 0.8)
            depth_total = max(50.0, random.gauss(600.0, 220.0))
            velocity = random.gauss(0.0, 0.03)
            volatility = max(0.0, random.gauss(0.02, 0.015))
            tts = random.randint(-5400, 86400)
            in_play = 1 if tts <= 0 else 0

            # Base market implied probability and decimal odds
            base_prob = clip01(random.betavariate(2.2, 2.8))
            odds = max(1.15, round(1.0 / base_prob * random.uniform(0.97, 1.05), 2))

            # Hidden residual edge: lower spread, strong positive imbalance, deep book,
            # and moderate volatility increase true win chance.
            residual = (
                0.28 * imbalance
                + 0.00035 * (depth_total - 600.0)
                - 1.9 * spread_mean
                + 0.75 * velocity
                - 1.1 * max(0.0, volatility - 0.03)
                + (0.06 if in_play == 0 and 600 <= tts <= 14400 else 0.0)
            )
            # Convert residual on logit scale.
            base_logit = math.log(base_prob / (1.0 - base_prob))
            p_true = clip01(1.0 / (1.0 + math.exp(-(base_logit + residual))))
            label = 1 if random.random() < p_true else 0

            writer.writerow(
                {
                    "timestamp": ts.isoformat(),
                    "base_prob": f"{base_prob:.6f}",
                    "odds": f"{odds:.2f}",
                    "label": str(label),
                    "spread_mean": f"{spread_mean:.6f}",
                    "imbalance": f"{imbalance:.6f}",
                    "depth_total_eur": f"{depth_total:.3f}",
                    "price_velocity": f"{velocity:.6f}",
                    "short_volatility": f"{volatility:.6f}",
                    "time_to_start_sec": str(tts),
                    "in_play": str(in_play),
                }
            )

    print(f"Wrote synthetic dataset: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
