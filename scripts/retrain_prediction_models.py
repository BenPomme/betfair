#!/usr/bin/env python3
"""
Retrain prediction models from online examples collected during paper trading.
Reads data/prediction/online_examples_*.jsonl, runs walk_forward_backtest,
overwrites model if Brier improves.
"""
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.predictive_model import (
    PredictionExample,
    PureLogitModel,
    ResidualLogitModel,
    walk_forward_backtest,
)


def load_examples(model_id: str) -> list[PredictionExample]:
    path = ROOT / f"data/prediction/online_examples_{model_id}.jsonl"
    if not path.exists():
        return []
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            features = {
                k: float(v) for k, v in raw.items()
                if k not in {"timestamp", "base_prob", "odds", "label", "model_id", "model_kind"}
            }
            examples.append(PredictionExample(
                timestamp=raw.get("timestamp", ""),
                base_prob=float(raw["base_prob"]),
                odds=float(raw["odds"]),
                label=int(raw["label"]),
                features=features,
            ))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return examples


def retrain_model(model_id: str, model_kind: str, min_examples: int = 50) -> dict:
    examples = load_examples(model_id)
    if len(examples) < min_examples:
        return {"model_id": model_id, "status": "skipped", "reason": f"only {len(examples)} examples (need {min_examples})"}

    model_dir = ROOT / "data/prediction/models"
    model_path = model_dir / f"{model_id}.json"

    # Load current model Brier if exists
    current_brier = None
    if model_path.exists():
        try:
            if model_kind == "residual_logit":
                current_model = ResidualLogitModel.load(str(model_path))
            elif model_kind == "pure_logit":
                current_model = PureLogitModel.load(str(model_path))
            else:
                return {"model_id": model_id, "status": "skipped", "reason": "implied_market has no trainable model"}
            # Evaluate current model on all examples
            brier_sum = 0.0
            for ex in examples:
                if model_kind == "residual_logit":
                    pred = current_model.predict_proba(ex.base_prob, ex.features)
                else:
                    pred = current_model.predict_proba(ex.features)
                brier_sum += (pred - ex.label) ** 2
            current_brier = brier_sum / len(examples)
        except Exception:
            current_brier = None

    # Train new model via walk-forward backtest
    feature_names = list(examples[0].features.keys()) if examples else []
    if model_kind == "residual_logit":
        new_model = ResidualLogitModel(feature_names=feature_names)
    elif model_kind == "pure_logit":
        new_model = PureLogitModel(feature_names=feature_names)
    else:
        return {"model_id": model_id, "status": "skipped", "reason": "implied_market"}

    try:
        results = walk_forward_backtest(new_model, examples, train_ratio=0.6, step=max(1, len(examples) // 20))
        new_brier = results.get("avg_brier", 1.0)
    except Exception as e:
        return {"model_id": model_id, "status": "error", "reason": str(e)}

    improved = current_brier is None or new_brier < current_brier
    if improved:
        model_dir.mkdir(parents=True, exist_ok=True)
        new_model.save(str(model_path))
        return {
            "model_id": model_id,
            "status": "updated",
            "new_brier": round(new_brier, 6),
            "old_brier": round(current_brier, 6) if current_brier is not None else None,
            "examples": len(examples),
        }
    return {
        "model_id": model_id,
        "status": "kept",
        "reason": "no improvement",
        "new_brier": round(new_brier, 6),
        "old_brier": round(current_brier, 6) if current_brier is not None else None,
    }


def main():
    models = [
        ("residual_logit_2", "residual_logit"),
        ("pure_logit_3", "pure_logit"),
    ]
    for model_id, model_kind in models:
        result = retrain_model(model_id, model_kind)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
