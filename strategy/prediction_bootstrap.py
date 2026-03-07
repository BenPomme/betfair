from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import config
from strategy.predictive_model import (
    HybridLogitModel,
    MarketCalibratedModel,
    PredictionExample,
)


def _configured_model_ids() -> Dict[str, str]:
    kinds = [x.strip() for x in str(config.PREDICTION_MODEL_KINDS).split(",") if x.strip()]
    return {f"{kind}_{idx+1}": kind for idx, kind in enumerate(kinds)}


def _load_examples(path: Path) -> List[PredictionExample]:
    rows: List[PredictionExample] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            rows.append(
                PredictionExample(
                    timestamp=str(raw["timestamp"]),
                    base_prob=float(raw["base_prob"]),
                    odds=float(raw["odds"]),
                    label=int(raw["label"]),
                    features={
                        str(k): float(v)
                        for k, v in raw.items()
                        if k not in {"timestamp", "base_prob", "odds", "label", "model_id", "model_kind", "source_model"}
                    },
                )
            )
        except Exception:
            continue
    rows.sort(key=lambda item: item.timestamp)
    return rows


def _dedupe_examples(examples: Iterable[PredictionExample]) -> List[PredictionExample]:
    out: List[PredictionExample] = []
    seen = set()
    for ex in examples:
        key = (
            ex.timestamp,
            round(float(ex.base_prob), 6),
            round(float(ex.odds), 4),
            int(ex.label),
            tuple(sorted((str(k), round(float(v), 6)) for k, v in ex.features.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    out.sort(key=lambda item: item.timestamp)
    return out


def _real_example_paths(input_dir: Path) -> List[Path]:
    configured = _configured_model_ids()
    paths: List[Path] = []
    for model_id in configured:
        path = input_dir / f"online_examples_{model_id}.jsonl"
        if path.exists():
            paths.append(path)
    return paths


def pooled_examples(input_dir: str = "data/prediction") -> List[PredictionExample]:
    root = Path(input_dir)
    combined: List[PredictionExample] = []
    for path in _real_example_paths(root):
        combined.extend(_load_examples(path))
    return _dedupe_examples(combined)


def bootstrap_challenger_models(
    *,
    input_dir: str = "data/prediction",
    model_dir: Optional[str] = None,
    force: bool = False,
    min_examples: int = 150,
) -> Dict[str, object]:
    configured = _configured_model_ids()
    targets = {
        model_id: kind
        for model_id, kind in configured.items()
        if kind in {"market_calibrated", "hybrid_logit"}
    }
    output_dir = Path(model_dir or config.PREDICTION_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = pooled_examples(input_dir=input_dir)
    summary = {
        "ok": True,
        "pooled_examples": len(examples),
        "bootstrapped": [],
        "skipped": [],
    }
    if len(examples) < int(min_examples):
        summary["ok"] = False
        summary["reason"] = "insufficient_examples"
        return summary

    feature_names = sorted({key for ex in examples for key in ex.features.keys()})
    train_examples = examples[:-max(50, int(len(examples) * 0.2))] if len(examples) > 200 else examples
    if not train_examples:
        train_examples = examples

    for model_id, kind in targets.items():
        out_path = output_dir / f"{model_id}.json"
        if out_path.exists() and not force:
            summary["skipped"].append({"model_id": model_id, "reason": "exists"})
            continue
        if kind == "market_calibrated":
            model = MarketCalibratedModel()
            model.fit(train_examples, epochs=12, lr=0.025)
        else:
            model = HybridLogitModel(feature_names)
            model.fit(train_examples, epochs=10, lr=0.02)
        model.save(str(out_path))
        summary["bootstrapped"].append({"model_id": model_id, "path": str(out_path)})
    return summary
