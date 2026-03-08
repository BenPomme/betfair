from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List

import config
from portfolio.types import ModelShadowAccount
from polymarket.features import feature_vector
from polymarket.utils import clamp, rolling_mean, sigmoid, to_float, utc_now_iso


_MODEL_DEFS = {
    "qm_coherence": {
        "feature_names": [
            "coherence_score",
            "orderbook_imbalance",
            "gamma_delta",
            "return_2m",
            "spread_bps",
        ],
        "weights": [1.1, 0.5, -0.7, 0.6, -0.15],
        "bias": 0.0,
    },
    "fold_basin": {
        "feature_names": [
            "folding_confidence",
            "basin_depth",
            "relaxation_speed",
            "energy_proxy",
            "spread_bps",
        ],
        "weights": [1.0, 0.7, 0.35, -0.9, -0.12],
        "bias": 0.0,
    },
    "hybrid_transition": {
        "feature_names": [
            "coherence_score",
            "folding_confidence",
            "orderbook_imbalance",
            "momentum_bias",
            "energy_proxy",
            "spread_bps",
        ],
        "weights": [0.8, 0.8, 0.4, 0.55, -0.85, -0.15],
        "bias": 0.0,
    },
}


def _brier(prediction: float, target: int) -> float:
    return (prediction - float(target)) ** 2


def _auc(points: Iterable[tuple[float, int]]) -> float | None:
    rows = [(float(score), int(target)) for score, target in points]
    positives = [score for score, target in rows if target == 1]
    negatives = [score for score, target in rows if target == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    pairs = 0.0
    for positive in positives:
        for negative in negatives:
            pairs += 1.0
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / pairs, 6) if pairs else None


@dataclass
class OnlineEdgeModel:
    model_id: str
    feature_names: List[str]
    weights: List[float]
    bias: float
    learning_rate: float = 0.08
    tracked_count: int = 0
    settled_count: int = 0
    update_count: int = 0
    shadow_trade_count: int = 0
    shadow_realized_pnl: float = 0.0
    shadow_current_balance: float = 0.0
    last_retrain_time: str | None = None
    last_retrain_result: str | None = None
    brier_sum: float = 0.0
    baseline_brier_sum: float = 0.0
    final_brier_sum: float = 0.0
    final_baseline_brier_sum: float = 0.0
    final_resolution_count: int = 0
    rolling_scores: Deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=200))
    rolling_brier_lifts: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def predict(self, features: Dict[str, Any]) -> float:
        vector = feature_vector(features, self.feature_names)
        score = self.bias
        for weight, value in zip(self.weights, vector):
            score += weight * value
        return round(clamp(sigmoid(score)), 6)

    def track(self, features: Dict[str, Any]) -> float:
        self.tracked_count += 1
        return self.predict(features)

    def settle(
        self,
        *,
        prediction: float,
        features: Dict[str, Any],
        target: int,
        net_return: float,
        baseline_probability: float | None = None,
        is_final: bool = False,
        stake_usd: float = 25.0,
        min_edge_prob: float = 0.02,
    ) -> None:
        self.settled_count += 1
        self.last_retrain_time = utc_now_iso()
        self.last_retrain_result = "accepted"
        baseline = clamp(baseline_probability if baseline_probability is not None else 0.5)
        self.brier_sum += _brier(prediction, target)
        self.baseline_brier_sum += _brier(baseline, target)
        self.rolling_scores.append((prediction, target))
        self.rolling_brier_lifts.append(_brier(baseline, target) - _brier(prediction, target))
        if is_final:
            self.final_resolution_count += 1
            self.final_brier_sum += _brier(prediction, target)
            self.final_baseline_brier_sum += _brier(baseline, target)
        signal_prob = prediction - 0.5
        if signal_prob >= min_edge_prob:
            self.shadow_trade_count += 1
            self.shadow_realized_pnl += float(net_return) * float(stake_usd)
        vector = feature_vector(features, self.feature_names)
        error = float(target) - prediction
        for idx, value in enumerate(vector):
            self.weights[idx] += self.learning_rate * error * value
        self.bias += self.learning_rate * error * 0.25
        self.update_count += 1

    def metrics(self) -> Dict[str, Any]:
        rolling_lift = rolling_mean(self.rolling_brier_lifts) if self.rolling_brier_lifts else 0.0
        current_auc = _auc(self.rolling_scores)
        final_lift = (
            (self.final_baseline_brier_sum - self.final_brier_sum) / self.final_resolution_count
            if self.final_resolution_count
            else 0.0
        )
        strict_gate_pass = (
            self.settled_count >= int(getattr(config, "POLYMARKET_QF_READINESS_MIN_LABELED", 250))
            and self.shadow_trade_count >= int(getattr(config, "POLYMARKET_QF_READINESS_MIN_CLOSED_TRADES", 50))
            and self.shadow_realized_pnl > 0
            and final_lift > 0
        )
        if self.settled_count < int(getattr(config, "POLYMARKET_QF_READINESS_MIN_LABELED", 250)):
            strict_gate_reason = "insufficient_labeled_examples"
        elif self.shadow_trade_count < int(getattr(config, "POLYMARKET_QF_READINESS_MIN_CLOSED_TRADES", 50)):
            strict_gate_reason = "insufficient_closed_trades"
        elif self.shadow_realized_pnl <= 0:
            strict_gate_reason = "negative_shadow_pnl"
        elif final_lift <= 0:
            strict_gate_reason = "negative_calibration_lift"
        else:
            strict_gate_reason = ""
        return {
            "model_kind": self.model_id,
            "learning_tracked": self.tracked_count,
            "learning_settled": self.settled_count,
            "model_updates": self.update_count,
            "current_auc": current_auc,
            "recent_learning_brier_lift": round(rolling_lift, 6),
            "rolling_200": {"brier_lift_abs": round(rolling_lift, 6)},
            "strict_gate_pass": strict_gate_pass,
            "strict_gate_reason": strict_gate_reason,
            "last_retrain_time": self.last_retrain_time,
            "last_retrain_result": self.last_retrain_result,
            "final_resolution_count": self.final_resolution_count,
            "final_brier_lift": round(final_lift, 6),
            "shadow_trade_count": self.shadow_trade_count,
        }


class QuantumFoldModelLeague:
    def __init__(self, portfolio_id: str, starting_balance: float) -> None:
        self.portfolio_id = portfolio_id
        self.starting_balance = float(starting_balance)
        self.models: Dict[str, OnlineEdgeModel] = {
            model_id: OnlineEdgeModel(
                model_id=model_id,
                feature_names=list(spec["feature_names"]),
                weights=[float(value) for value in spec["weights"]],
                bias=float(spec["bias"]),
                shadow_current_balance=float(starting_balance),
            )
            for model_id, spec in _MODEL_DEFS.items()
        }

    def track_example(self, features: Dict[str, Any]) -> Dict[str, float]:
        return {
            model_id: model.track(features)
            for model_id, model in self.models.items()
        }

    def predict_all(self, features: Dict[str, Any]) -> Dict[str, float]:
        return {
            model_id: model.predict(features)
            for model_id, model in self.models.items()
        }

    def settle_labels(self, labels: Iterable[Dict[str, Any]], primary_horizon: int) -> None:
        for label in labels:
            if not isinstance(label, dict):
                continue
            features = dict(label.get("features") or {})
            target = int(label.get("target", 0) or 0)
            net_return = to_float(label.get("net_return"), 0.0)
            predictions = dict(label.get("model_predictions") or {})
            baseline_probability = label.get("baseline_probability")
            is_primary = str(label.get("horizon_label")) == f"{int(primary_horizon)}s"
            is_final = str(label.get("horizon_label")) == "final"
            if not (is_primary or is_final):
                continue
            for model_id, model in self.models.items():
                prediction = clamp(to_float(predictions.get(model_id), model.predict(features)))
                model.settle(
                    prediction=prediction,
                    features=features,
                    target=target,
                    net_return=net_return,
                    baseline_probability=to_float(baseline_probability, 0.5) if baseline_probability is not None else None,
                    is_final=is_final,
                )
                model.shadow_current_balance = self.starting_balance + model.shadow_realized_pnl

    def build_accounts(self) -> List[ModelShadowAccount]:
        accounts: List[ModelShadowAccount] = []
        for model_id, model in self.models.items():
            metrics = model.metrics()
            accounts.append(
                ModelShadowAccount(
                    portfolio_id=self.portfolio_id,
                    model_id=model_id,
                    shadow_starting_balance=self.starting_balance,
                    shadow_current_balance=round(model.shadow_current_balance, 6),
                    shadow_realized_pnl=round(model.shadow_realized_pnl, 6),
                    shadow_roi_pct=round((model.shadow_realized_pnl / self.starting_balance) * 100.0, 6) if self.starting_balance else 0.0,
                    settled_count=model.settled_count,
                    metrics=metrics,
                    selected_for_execution=(model_id == "hybrid_transition"),
                )
            )
        return accounts

    def summary(self) -> Dict[str, Any]:
        ranked = sorted(
            (
                {
                    "model_id": model_id,
                    "shadow_realized_pnl": round(model.shadow_realized_pnl, 6),
                    "settled_count": model.settled_count,
                    "recent_learning_brier_lift": round(rolling_mean(model.rolling_brier_lifts), 6) if model.rolling_brier_lifts else 0.0,
                    "strict_gate_pass": model.metrics()["strict_gate_pass"],
                }
                for model_id, model in self.models.items()
            ),
            key=lambda item: (item["strict_gate_pass"], item["shadow_realized_pnl"], item["recent_learning_brier_lift"]),
            reverse=True,
        )
        return {
            "leader_model_id": ranked[0]["model_id"] if ranked else None,
            "ranked_models": ranked,
        }
