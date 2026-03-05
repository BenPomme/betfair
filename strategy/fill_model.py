"""
Train lightweight fill probability model from candidate logs.

Usage:
  python -m strategy.fill_model --input-dir data/candidates --output data/models/fill_model_v1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


FEATURES = [
    "spread_mean",
    "depth_total_eur",
    "short_volatility",
    "time_to_start_sec",
    "in_play",
    "total_stake_eur",
]


@dataclass
class Example:
    x: Dict[str, float]
    y: int


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _load_rows(input_dir: str) -> List[dict]:
    rows: List[dict] = []
    for fp in sorted(glob.glob(str(Path(input_dir) / "*.jsonl"))):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _build_examples(rows: List[dict]) -> List[Example]:
    out: List[Example] = []
    for r in rows:
        reason = str(r.get("reason", ""))
        executed = bool(r.get("executed", False))
        if executed:
            y = 1
        elif reason in {"risk_blocked", "fill_prob_below_min", "decision_skip"}:
            y = 0
        else:
            continue
        try:
            x = {
                "spread_mean": float(r.get("overround_lay", 0.0)),
                "depth_total_eur": float(r.get("overround_back", 0.0)),
                "short_volatility": float(r.get("edge_score", 0.0)),
                "time_to_start_sec": float(r.get("time_to_start_sec", 0.0)),
                "in_play": 1.0 if float(r.get("time_to_start_sec", 1.0)) <= 0 else 0.0,
                "total_stake_eur": float(r.get("total_stake_eur", 0.0)),
            }
        except Exception:
            continue
        out.append(Example(x=x, y=y))
    return out


def _train(examples: List[Example], epochs: int = 10, lr: float = 0.03, l2: float = 1e-4) -> Dict[str, float]:
    weights = {k: 0.0 for k in FEATURES}
    bias = 0.0
    for _ in range(max(1, epochs)):
        for ex in examples:
            z = bias + sum(weights[k] * ex.x.get(k, 0.0) for k in FEATURES)
            p = _sigmoid(z)
            err = p - ex.y
            bias -= lr * err
            for k in FEATURES:
                v = ex.x.get(k, 0.0)
                grad = err * v + l2 * weights[k]
                weights[k] -= lr * grad
    return {"bias": bias, **weights}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/candidates")
    parser.add_argument("--output", default="data/models/fill_model_v1.json")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    rows = _load_rows(args.input_dir)
    if not rows:
        print("No candidate rows found.")
        return 1
    examples = _build_examples(rows)
    if len(examples) < args.min_samples:
        print(f"Not enough samples: {len(examples)} < {args.min_samples}")
        return 1

    model = _train(examples)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved fill model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
