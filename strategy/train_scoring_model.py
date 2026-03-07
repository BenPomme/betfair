"""
Train scoring linear model from candidate logger JSONL data.

Usage:
  python -m strategy.train_scoring_model --input-dir data/candidates --output data/models/scoring_linear_v2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


FEATURE_MAP = {
    "roi": ("net_roi_pct",),
    "profit": ("net_profit_eur",),
    "depth": ("depth_total_eur", "overround_back"),
    "spread": ("spread_mean", "overround_lay"),
    "volatility": ("short_volatility", "fill_prob"),
    "edge_score": ("edge_score",),
}

EXCLUDE_REASONS = {"no_arb_after_filters", "stale_or_missing_snapshot"}

BOOTSTRAP_PRIOR = {
    "bias": -0.10,
    "roi": 1.20,
    "profit": 0.18,
    "depth": 0.00001,
    "spread": -0.40,
    "volatility": -0.25,
    "edge_score": 0.45,
}


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


def _load_records(input_dir: str) -> List[dict]:
    files = sorted(glob.glob(str(Path(input_dir) / "*.jsonl")))
    records: List[dict] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _build_examples(records: List[dict]) -> List[Example]:
    out: List[Example] = []
    for r in records:
        if r.get("reason") in EXCLUDE_REASONS:
            continue
        if "executed" not in r:
            continue
        x = {}
        missing = False
        for k_out, source_keys in FEATURE_MAP.items():
            selected = None
            for source_key in source_keys:
                if source_key in r:
                    selected = source_key
                    break
            if selected is None:
                missing = True
                break
            x[k_out] = float(r[selected])
        if missing:
            continue
        y = 1 if bool(r.get("executed")) else 0
        out.append(Example(x=x, y=y))
    return out


def train_from_logs(
    input_dir: str = "data/candidates",
    output: str = "data/models/scoring_linear_v2.json",
    min_samples: int = 100,
) -> Dict[str, object]:
    records = _load_records(input_dir)
    if not records:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(BOOTSTRAP_PRIOR, indent=2), encoding="utf-8")
        return {"ok": True, "reason": "bootstrap_prior", "output": str(out_path), "samples": 0}
    examples = _build_examples(records)
    if len(examples) < min_samples:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(BOOTSTRAP_PRIOR, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "reason": "bootstrap_prior",
            "output": str(out_path),
            "samples": len(examples),
            "required": min_samples,
        }

    model = _train_sgd(examples)
    out = {
        "bias": model["bias"],
        "roi": model["roi"],
        "profit": model["profit"],
        "depth": model["depth"],
        "spread": model["spread"],
        "volatility": model["volatility"],
        "edge_score": model["edge_score"],
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    acc, brier, cm = _walk_forward(examples)
    return {
        "ok": True,
        "output": str(out_path),
        "samples": len(examples),
        "walk_forward_accuracy": round(acc, 4),
        "walk_forward_brier": round(brier, 4),
        "confusion_matrix": cm,
    }


def _train_sgd(examples: List[Example], epochs: int = 12, lr: float = 0.03, l2: float = 1e-4) -> Dict[str, float]:
    weights = {k: 0.0 for k in FEATURE_MAP.keys()}
    bias = 0.0
    for _ in range(max(1, epochs)):
        for ex in examples:
            z = bias
            for k, v in ex.x.items():
                z += weights[k] * v
            p = _sigmoid(z)
            err = p - ex.y
            bias -= lr * err
            for k, v in ex.x.items():
                grad = err * v + l2 * weights[k]
                weights[k] -= lr * grad
    return {"bias": bias, **weights}


def _predict(model: Dict[str, float], x: Dict[str, float]) -> float:
    z = float(model.get("bias", 0.0))
    for k, v in x.items():
        z += float(model.get(k, 0.0)) * v
    return _sigmoid(z)


def _evaluate(model: Dict[str, float], examples: List[Example]) -> Tuple[float, float, Dict[str, int]]:
    if not examples:
        return 0.0, 0.0, {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    correct = 0
    brier = 0.0
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for ex in examples:
        p = _predict(model, ex.x)
        pred = 1 if p >= 0.5 else 0
        if pred == ex.y:
            correct += 1
        brier += (p - ex.y) ** 2
        if pred == 1 and ex.y == 1:
            cm["tp"] += 1
        elif pred == 1 and ex.y == 0:
            cm["fp"] += 1
        elif pred == 0 and ex.y == 0:
            cm["tn"] += 1
        else:
            cm["fn"] += 1
    return correct / len(examples), brier / len(examples), cm


def _walk_forward(examples: List[Example], train_ratio: float = 0.7) -> Tuple[float, float, Dict[str, int]]:
    n = len(examples)
    if n < 2:
        return 0.0, 0.0, {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    split = int(n * train_ratio)
    split = max(1, min(n - 1, split))
    model = _train_sgd(examples[:split])
    return _evaluate(model, examples[split:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/candidates")
    parser.add_argument("--output", default="data/models/scoring_linear_v2.json")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    result = train_from_logs(args.input_dir, args.output, args.min_samples)
    if not result.get("ok"):
        reason = result.get("reason", "unknown")
        if reason == "insufficient_samples":
            print(f"Not enough samples: {result.get('samples')} < {result.get('required')}")
        elif reason == "no_candidate_records":
            print("No candidate records found.")
        else:
            print(f"Training failed: {reason}")
        return 1

    print(f"Samples: {result['samples']}")
    print(f"Saved model: {result['output']}")
    print(f"Walk-forward accuracy: {result['walk_forward_accuracy']:.4f}")
    print(f"Walk-forward brier: {result['walk_forward_brier']:.4f}")
    print(f"Confusion matrix: {result['confusion_matrix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
