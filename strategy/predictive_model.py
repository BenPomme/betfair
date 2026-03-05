"""
Predictive sports model:
- Base signal from market implied probability
- ML residual learned from microstructure/context features

This is intentionally standalone from arbitrage execution so it can be tested
in paper mode without touching financially critical live paths.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _clip_prob(p: float) -> float:
    if p < 1e-6:
        return 1e-6
    if p > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return p


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = _clip_prob(p)
    return math.log(p / (1.0 - p))

def _transform_feature(name: str, value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    scales = {
        "depth_total_eur": 1000.0,
        "time_to_start_sec": 43200.0,
        "selection_count": 10.0,
        "weighted_spread": 2.0,
        "spread_mean": 2.0,
        "price_range": 5.0,
        "volume_momentum": 1000.0,
        "lay_back_ratio": 3.0,
    }
    scale = scales.get(name, 1.0)
    x = value / scale
    if x < -3.0:
        return -3.0
    if x > 3.0:
        return 3.0
    return x

def _clip_contribution(v: float, cap: float = 1.0) -> float:
    if v < -cap:
        return -cap
    if v > cap:
        return cap
    return v


@dataclass(frozen=True)
class PredictionExample:
    """One labeled example for model training/evaluation."""
    timestamp: str
    base_prob: float
    odds: float
    label: int  # 1 if predicted selection wins, else 0
    features: Dict[str, float]


@dataclass(frozen=True)
class PredictiveMetrics:
    brier: float
    logloss: float
    accuracy: float
    bets: int
    roi: float
    pnl_units: float


class ResidualLogitModel:
    """
    Learn a residual on top of market implied probability:
      p_hat = sigmoid(logit(base_prob) + bias + w·x)
    """

    def __init__(self, feature_names: Iterable[str]):
        self.feature_names = list(feature_names)
        self.bias = 0.0
        self.weights = {name: 0.0 for name in self.feature_names}

    def predict_proba(self, base_prob: float, features: Dict[str, float]) -> float:
        z = _logit(base_prob) + self.bias
        for name in self.feature_names:
            x = _transform_feature(name, float(features.get(name, 0.0)))
            z += _clip_contribution(self.weights[name] * x)
        return _clip_prob(_sigmoid(z))

    def fit(
        self,
        examples: List[PredictionExample],
        epochs: int = 8,
        lr: float = 0.05,
        l2: float = 1e-4,
    ) -> None:
        if not examples:
            return
        for _ in range(max(1, epochs)):
            for ex in examples:
                p = self.predict_proba(ex.base_prob, ex.features)
                err = p - float(ex.label)
                # Bias gradient
                self.bias -= lr * err
                # Weight gradient
                for name in self.feature_names:
                    x = _transform_feature(name, float(ex.features.get(name, 0.0)))
                    grad = err * x + l2 * self.weights[name]
                    self.weights[name] -= lr * grad

    def to_dict(self) -> dict:
        return {
            "model_type": "residual_logit_v1",
            "feature_names": self.feature_names,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ResidualLogitModel":
        feature_names = payload.get("feature_names", [])
        model = cls(feature_names=feature_names)
        model.bias = float(payload.get("bias", 0.0))
        model.weights = {name: float(payload.get("weights", {}).get(name, 0.0)) for name in feature_names}
        return model

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ResidualLogitModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class PureLogitModel:
    """
    Logistic model that predicts from microstructure/context features only:
      p_hat = sigmoid(bias + w·x)
    """

    def __init__(self, feature_names: Iterable[str]):
        self.feature_names = list(feature_names)
        self.bias = 0.0
        self.weights = {name: 0.0 for name in self.feature_names}

    def predict_proba(self, features: Dict[str, float]) -> float:
        z = self.bias
        for name in self.feature_names:
            x = _transform_feature(name, float(features.get(name, 0.0)))
            z += _clip_contribution(self.weights[name] * x)
        return _clip_prob(_sigmoid(z))

    def fit(
        self,
        examples: List[PredictionExample],
        epochs: int = 8,
        lr: float = 0.03,
        l2: float = 1e-4,
    ) -> None:
        if not examples:
            return
        for _ in range(max(1, epochs)):
            for ex in examples:
                p = self.predict_proba(ex.features)
                err = p - float(ex.label)
                self.bias -= lr * err
                for name in self.feature_names:
                    x = _transform_feature(name, float(ex.features.get(name, 0.0)))
                    grad = err * x + l2 * self.weights[name]
                    self.weights[name] -= lr * grad

    def to_dict(self) -> dict:
        return {
            "model_type": "pure_logit_v1",
            "feature_names": self.feature_names,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PureLogitModel":
        feature_names = payload.get("feature_names", [])
        model = cls(feature_names=feature_names)
        model.bias = float(payload.get("bias", 0.0))
        model.weights = {name: float(payload.get("weights", {}).get(name, 0.0)) for name in feature_names}
        return model

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "PureLogitModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def evaluate_predictions(
    probs: List[float],
    labels: List[int],
    odds: List[float],
    edge_threshold: float = 0.02,
    stake: float = 1.0,
) -> PredictiveMetrics:
    n = max(1, len(labels))
    brier = 0.0
    logloss = 0.0
    correct = 0
    bets = 0
    pnl = 0.0

    for p, y, o in zip(probs, labels, odds):
        p = _clip_prob(float(p))
        y = int(y)
        o = float(o)
        brier += (p - y) ** 2
        logloss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        pred_y = 1 if p >= 0.5 else 0
        if pred_y == y:
            correct += 1

        implied = 1.0 / max(o, 1.01)
        if p - implied >= edge_threshold:
            bets += 1
            pnl -= stake
            if y == 1:
                pnl += stake * o

    roi = (pnl / (bets * stake)) if bets > 0 else 0.0
    return PredictiveMetrics(
        brier=brier / n,
        logloss=logloss / n,
        accuracy=correct / n,
        bets=bets,
        roi=roi,
        pnl_units=pnl,
    )


def walk_forward_backtest(
    examples: List[PredictionExample],
    train_window: int = 400,
    test_window: int = 100,
    epochs: int = 8,
    lr: float = 0.05,
    edge_threshold: float = 0.02,
    stake: float = 1.0,
) -> Tuple[PredictiveMetrics, List[float]]:
    """
    Walk-forward:
    - train on [i : i+train_window)
    - test on [i+train_window : i+train_window+test_window)
    - repeat with expanding cursor
    """
    if len(examples) < train_window + test_window:
        empty = PredictiveMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0)
        return empty, []

    feature_names = sorted({k for ex in examples for k in ex.features.keys()})
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_odds: List[float] = []

    i = 0
    while i + train_window + test_window <= len(examples):
        train = examples[i : i + train_window]
        test = examples[i + train_window : i + train_window + test_window]
        model = ResidualLogitModel(feature_names=feature_names)
        model.fit(train, epochs=epochs, lr=lr)
        for ex in test:
            p = model.predict_proba(ex.base_prob, ex.features)
            all_probs.append(p)
            all_labels.append(ex.label)
            all_odds.append(ex.odds)
        i += test_window

    metrics = evaluate_predictions(
        probs=all_probs,
        labels=all_labels,
        odds=all_odds,
        edge_threshold=edge_threshold,
        stake=stake,
    )
    return metrics, all_probs
