"""
Backtest-driven execution gate for Betfair prediction models.

Purpose:
- keep weak models learning in shadow mode
- only let models place paper bets when walk-forward evidence is positive
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import config
from strategy.predictive_model import (
    PredictionExample,
    PredictiveMetrics,
    PureLogitModel,
    ResidualLogitModel,
    evaluate_predictions,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelPolicy:
    model_id: str
    model_kind: str
    mode: str
    reason: str
    train_examples: int
    test_examples: int
    bets: int
    roi: float
    pnl_units: float
    brier: float
    baseline_brier: float
    brier_lift: float
    stake_multiplier: float
    edge_threshold: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_CACHE_PATH: Optional[Path] = None
_CACHE_MTIME: Optional[float] = None
_CACHE_PAYLOAD: Dict[str, object] = {}


def _artifact_path() -> Path:
    return Path(config.PREDICTION_POLICY_GATE_PATH)


def _iter_example_paths(input_dir: str) -> Iterable[Path]:
    root = Path(input_dir)
    return sorted(root.glob("online_examples_*.jsonl"))


def _infer_model_id(path: Path) -> str:
    return path.stem.replace("online_examples_", "")


def _state_payload(model_id: str) -> Dict[str, object]:
    state_path = Path(config.PREDICTION_STATE_DIR) / f"{model_id}.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_examples(path: Path) -> List[PredictionExample]:
    rows: List[PredictionExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        features = {
            str(k): float(v)
            for k, v in raw.items()
            if k not in {"timestamp", "base_prob", "odds", "label", "model_id", "model_kind", "source_model"}
        }
        try:
            rows.append(
                PredictionExample(
                    timestamp=str(raw["timestamp"]),
                    base_prob=float(raw["base_prob"]),
                    odds=float(raw["odds"]),
                    label=int(raw["label"]),
                    features=features,
                )
            )
        except Exception:
            continue
    rows.sort(key=lambda item: item.timestamp)
    return rows


def _walk_forward_metrics(
    *,
    model_kind: str,
    examples: List[PredictionExample],
    train_examples: int,
    edge_threshold: float,
) -> tuple[PredictiveMetrics, float]:
    test = examples[train_examples:]
    if not test:
        empty = PredictiveMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0)
        return empty, 0.0
    baseline_brier = sum((ex.base_prob - ex.label) ** 2 for ex in test) / len(test)
    if model_kind == "implied_market":
        metrics = evaluate_predictions(
            probs=[ex.base_prob for ex in test],
            labels=[ex.label for ex in test],
            odds=[ex.odds for ex in test],
            edge_threshold=edge_threshold,
            stake=1.0,
        )
        return metrics, baseline_brier

    feature_names = sorted({key for ex in examples for key in ex.features.keys()})
    if model_kind == "pure_logit":
        model = PureLogitModel(feature_names)
        model.fit(examples[:train_examples], epochs=4, lr=0.02)
    else:
        model = ResidualLogitModel(feature_names)
        model.fit(examples[:train_examples], epochs=4, lr=0.02)

    probs: List[float] = []
    labels: List[int] = []
    odds: List[float] = []
    for ex in test:
        if model_kind == "pure_logit":
            prob = model.predict_proba(ex.features)
        else:
            prob = model.predict_proba(ex.base_prob, ex.features)
        probs.append(prob)
        labels.append(ex.label)
        odds.append(ex.odds)
        model.fit([ex], epochs=1, lr=0.02)
    metrics = evaluate_predictions(
        probs=probs,
        labels=labels,
        odds=odds,
        edge_threshold=edge_threshold,
        stake=1.0,
    )
    return metrics, baseline_brier


def _stake_multiplier_from_metrics(metrics: PredictiveMetrics, brier_lift: float) -> float:
    if metrics.roi <= 0 or brier_lift <= 0:
        return 0.0
    if metrics.roi >= 0.05 and brier_lift >= 0.01:
        return 1.0
    if metrics.roi >= 0.02:
        return 0.75
    return 0.5


def train_from_examples(
    input_dir: str = "data/prediction",
    output: Optional[str] = None,
) -> Dict[str, object]:
    out_path = Path(output or config.PREDICTION_POLICY_GATE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "created_at": _utc_now_iso(),
        "model_policies": {},
        "summary": {
            "models_seen": 0,
            "models_execute": 0,
            "models_shadow_only": 0,
        },
    }

    min_examples = max(50, int(config.PREDICTION_POLICY_GATE_MIN_EXAMPLES))
    min_test_examples = max(20, int(config.PREDICTION_POLICY_GATE_MIN_TEST_EXAMPLES))
    train_fraction = float(config.PREDICTION_POLICY_GATE_TRAIN_FRACTION)
    min_roi = float(config.PREDICTION_POLICY_GATE_MIN_ROI)
    min_brier_lift = float(config.PREDICTION_POLICY_GATE_MIN_BRIER_LIFT)
    min_test_bets = max(1, int(config.PREDICTION_POLICY_GATE_MIN_TEST_BETS))

    for path in _iter_example_paths(input_dir):
        model_id = _infer_model_id(path)
        state = _state_payload(model_id)
        model_kind = str(state.get("model_kind", "unknown") or "unknown")
        examples = _load_examples(path)
        payload["summary"]["models_seen"] = int(payload["summary"]["models_seen"]) + 1
        edge_threshold = float(state.get("min_edge", config.PREDICTION_MIN_EDGE))

        if model_kind == "unknown":
            policy = ModelPolicy(
                model_id=model_id,
                model_kind=model_kind,
                mode="shadow_only",
                reason="missing_model_kind",
                train_examples=0,
                test_examples=0,
                bets=0,
                roi=0.0,
                pnl_units=0.0,
                brier=0.0,
                baseline_brier=0.0,
                brier_lift=0.0,
                stake_multiplier=0.0,
                edge_threshold=edge_threshold,
            )
        elif len(examples) < min_examples:
            policy = ModelPolicy(
                model_id=model_id,
                model_kind=model_kind,
                mode="shadow_only",
                reason="insufficient_examples",
                train_examples=len(examples),
                test_examples=0,
                bets=0,
                roi=0.0,
                pnl_units=0.0,
                brier=0.0,
                baseline_brier=0.0,
                brier_lift=0.0,
                stake_multiplier=0.0,
                edge_threshold=edge_threshold,
            )
        else:
            train_examples = max(50, int(len(examples) * train_fraction))
            test_examples = len(examples) - train_examples
            if test_examples < min_test_examples:
                train_examples = max(50, len(examples) - min_test_examples)
                test_examples = len(examples) - train_examples
            metrics, baseline_brier = _walk_forward_metrics(
                model_kind=model_kind,
                examples=examples,
                train_examples=train_examples,
                edge_threshold=edge_threshold,
            )
            brier_lift = baseline_brier - metrics.brier
            mode = "shadow_only"
            reason = "negative_walk_forward"
            if (
                test_examples >= min_test_examples
                and metrics.bets >= min_test_bets
                and metrics.roi >= min_roi
                and brier_lift >= min_brier_lift
            ):
                mode = "execute"
                reason = "walk_forward_pass"
            elif metrics.bets < min_test_bets:
                reason = "insufficient_test_bets"
            elif brier_lift < min_brier_lift:
                reason = "negative_brier_lift"
            elif metrics.roi < min_roi:
                reason = "negative_roi"
            policy = ModelPolicy(
                model_id=model_id,
                model_kind=model_kind,
                mode=mode,
                reason=reason,
                train_examples=train_examples,
                test_examples=test_examples,
                bets=metrics.bets,
                roi=float(metrics.roi),
                pnl_units=float(metrics.pnl_units),
                brier=float(metrics.brier),
                baseline_brier=float(baseline_brier),
                brier_lift=float(brier_lift),
                stake_multiplier=_stake_multiplier_from_metrics(metrics, brier_lift),
                edge_threshold=edge_threshold,
            )

        payload["model_policies"][model_id] = policy.to_dict()
        if policy.mode == "execute":
            payload["summary"]["models_execute"] = int(payload["summary"]["models_execute"]) + 1
        else:
            payload["summary"]["models_shadow_only"] = int(payload["summary"]["models_shadow_only"]) + 1

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ok": True, "output": str(out_path), "summary": payload["summary"]}


def _load_artifact() -> Dict[str, object]:
    path = _artifact_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _refresh_cache() -> None:
    global _CACHE_PATH, _CACHE_MTIME, _CACHE_PAYLOAD
    path = _artifact_path()
    mtime = path.stat().st_mtime if path.exists() else None
    if _CACHE_PATH != path or _CACHE_MTIME != mtime:
        _CACHE_PATH = path
        _CACHE_MTIME = mtime
        _CACHE_PAYLOAD = _load_artifact()


def get_model_policy(model_id: str) -> Dict[str, object]:
    if not bool(config.PREDICTION_POLICY_GATE_ENABLED):
        return {"mode": "execute", "reason": "policy_gate_disabled", "stake_multiplier": 1.0}
    _refresh_cache()
    model_policies = dict(_CACHE_PAYLOAD.get("model_policies") or {})
    return dict(model_policies.get(model_id) or {"mode": "execute", "reason": "policy_gate_missing", "stake_multiplier": 1.0})
