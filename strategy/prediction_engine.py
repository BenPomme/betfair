"""
Multi-model online prediction paper accounts.

Each model runs its own fake bankroll in parallel so performance can be compared
live in the dashboard.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

import config
from core.types import PriceSnapshot
from data.clv_tracker import CLVTracker
from strategy.features import build_market_microstructure
from strategy.prediction_policy_gate import get_model_policy
from strategy.predictive_model import PredictionExample, PureLogitModel, ResidualLogitModel

FEATURE_NAMES = [
    "spread_mean",
    "imbalance",
    "depth_total_eur",
    "price_velocity",
    "short_volatility",
    "time_to_start_sec",
    "in_play",
    "weighted_spread",
    "lay_back_ratio",
    "top_of_book_concentration",
    "selection_count",
    "volume_momentum",
    "back_lay_crossover",
    "overround_distance",
    "depth_ratio_top3",
    "price_range",
]

RECENT_LIFT_WINDOW = 200
ROLLING_WINDOWS = (50, 100, 200)


@dataclass(frozen=True)
class _PendingBet:
    bet_id: str
    market_id: str
    selection_id: str
    selection_name: str
    event_name: str
    entry_odds: float
    stake: float
    base_prob: float
    predicted_prob: float
    features: Dict[str, float]


@dataclass(frozen=True)
class _LearningCandidate:
    market_id: str
    selection_id: str
    selection_name: str
    event_name: str
    entry_odds: float
    base_prob: float
    predicted_prob: float
    features: Dict[str, float]


class OnlinePredictionEngine:
    """
    One prediction model + one fake account.

    Supported model kinds:
    - implied_market: market-implied probability only (control)
    - residual_logit: online residual logistic model
    - pure_logit: online logistic model on microstructure features only
    """

    def __init__(
        self,
        model_id: str,
        model_kind: str,
        initial_balance_eur: float,
        stake_fraction: float,
        min_stake_eur: float,
        max_stake_eur: float,
        min_edge: float,
        min_liquidity_eur: float,
        model_path: str,
        save_every: int = 25,
        clv_tracker: Optional[CLVTracker] = None,
        state_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_kind = model_kind
        self.initial_balance = float(initial_balance_eur)
        self.balance = float(initial_balance_eur)
        self.stake_fraction = float(stake_fraction)
        self.min_stake = float(min_stake_eur)
        self.max_stake = float(max_stake_eur)
        self.min_edge = float(min_edge)
        self.min_liquidity = float(min_liquidity_eur)
        self.model_path = Path(model_path)
        self.save_every = max(1, int(save_every))

        self.total_bets = 0
        self.settled_bets = 0
        self.wins = 0
        self.losses = 0
        self.voids = 0
        self.resets = 0
        self.total_pnl = 0.0
        self.update_count = 0
        self.brier_sum = 0.0
        self.last_edge = 0.0
        self.last_prediction = 0.0
        self.learning_tracked = 0
        self.learning_settled = 0
        self.learning_model_updates = 0
        self.learning_brier_sum = 0.0
        self.baseline_brier_sum = 0.0
        self.learning_baseline_brier_sum = 0.0
        self._recent_brier_lift = deque(maxlen=RECENT_LIFT_WINDOW)
        self._recent_learning_brier_lift = deque(maxlen=RECENT_LIFT_WINDOW)
        self._settled_history = deque(maxlen=500)
        self._prediction_history = deque(maxlen=max(300, int(config.PREDICTION_SATURATION_WINDOW)))
        self._feature_drift_stats: Dict[str, Dict[str, float]] = {}
        self._update_rejections: deque = deque(maxlen=200)
        self._consecutive_gate_passes = 0
        self._last_gate_state: Optional[bool] = None

        self._pending: Dict[Tuple[str, str], _PendingBet] = {}
        self._learning_candidates: Dict[str, _LearningCandidate] = {}
        self._prev_snapshots: Dict[str, PriceSnapshot] = {}
        self._examples_log = Path(f"data/prediction/online_examples_{self.model_id}.jsonl")
        self._examples_log.parent.mkdir(parents=True, exist_ok=True)
        self._clv_tracker = clv_tracker
        self._state_path = Path(state_path) if state_path else None
        self._experiment_log_path = Path(config.PREDICTION_EXPERIMENT_LOG_PATH)
        self._enforcement_mode = str(config.PREDICTION_GATE_ENFORCEMENT_MODE or "observe").lower()
        self._pass_streak_required = max(1, int(config.PREDICTION_GATE_PASS_STREAK_REQUIRED))

        self.model = self._load_or_init_model()
        self._load_state()

    def has_pending_market(self, market_id: str) -> bool:
        if market_id in self._learning_candidates:
            return True
        return any(b.market_id == market_id for b in self._pending.values())

    def pending_market_ids(self) -> List[str]:
        mids = {b.market_id for b in self._pending.values()}
        mids.update(self._learning_candidates.keys())
        return sorted(mids)

    def _load_or_init_model(self):
        if self.model_kind == "implied_market":
            return None
        if self.model_kind == "residual_logit":
            if self.model_path.exists():
                try:
                    return ResidualLogitModel.load(str(self.model_path))
                except Exception:
                    pass
            return ResidualLogitModel(feature_names=FEATURE_NAMES)
        if self.model_kind == "pure_logit":
            if self.model_path.exists():
                try:
                    return PureLogitModel.load(str(self.model_path))
                except Exception:
                    pass
            return PureLogitModel(feature_names=FEATURE_NAMES)
        raise ValueError(f"Unsupported model_kind: {self.model_kind}")

    def _maybe_save_model(self) -> None:
        if self.model is None:
            return
        if self.update_count % self.save_every == 0:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.model_path))

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "balance": self.balance,
            "total_bets": self.total_bets,
            "settled_bets": self.settled_bets,
            "wins": self.wins,
            "losses": self.losses,
            "voids": self.voids,
            "resets": self.resets,
            "total_pnl": self.total_pnl,
            "update_count": self.update_count,
            "brier_sum": self.brier_sum,
            "last_edge": self.last_edge,
            "last_prediction": self.last_prediction,
            "learning_tracked": self.learning_tracked,
            "learning_settled": self.learning_settled,
            "learning_model_updates": self.learning_model_updates,
            "learning_brier_sum": self.learning_brier_sum,
            "baseline_brier_sum": self.baseline_brier_sum,
            "learning_baseline_brier_sum": self.learning_baseline_brier_sum,
            "recent_brier_lift": list(self._recent_brier_lift),
            "recent_learning_brier_lift": list(self._recent_learning_brier_lift),
            "settled_history": list(self._settled_history),
            "prediction_history": list(self._prediction_history),
            "feature_drift_stats": self._feature_drift_stats,
            "update_rejections": list(self._update_rejections),
            "consecutive_gate_passes": self._consecutive_gate_passes,
            "last_gate_state": self._last_gate_state,
            "stake_fraction": self.stake_fraction,
            "min_edge": self.min_edge,
            "pending": [
                {
                    "market_id": b.market_id,
                    "selection_id": b.selection_id,
                    "bet_id": b.bet_id,
                    "selection_name": b.selection_name,
                    "event_name": b.event_name,
                    "entry_odds": b.entry_odds,
                    "stake": b.stake,
                    "base_prob": b.base_prob,
                    "predicted_prob": b.predicted_prob,
                    "features": b.features,
                }
                for b in self._pending.values()
            ],
            "learning_candidates": [
                {
                    "market_id": c.market_id,
                    "selection_id": c.selection_id,
                    "selection_name": c.selection_name,
                    "event_name": c.event_name,
                    "entry_odds": c.entry_odds,
                    "base_prob": c.base_prob,
                    "predicted_prob": c.predicted_prob,
                    "features": c.features,
                }
                for c in self._learning_candidates.values()
            ],
        }
        self._state_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.balance = float(raw.get("balance", self.balance))
            self.total_bets = int(raw.get("total_bets", self.total_bets))
            self.settled_bets = int(raw.get("settled_bets", self.settled_bets))
            self.wins = int(raw.get("wins", self.wins))
            self.losses = int(raw.get("losses", self.losses))
            self.voids = int(raw.get("voids", self.voids))
            self.resets = int(raw.get("resets", self.resets))
            self.total_pnl = float(raw.get("total_pnl", self.total_pnl))
            self.update_count = int(raw.get("update_count", self.update_count))
            self.brier_sum = float(raw.get("brier_sum", self.brier_sum))
            self.last_edge = float(raw.get("last_edge", self.last_edge))
            self.last_prediction = float(raw.get("last_prediction", self.last_prediction))
            self.learning_tracked = int(raw.get("learning_tracked", self.learning_tracked))
            self.learning_settled = int(raw.get("learning_settled", self.learning_settled))
            self.learning_model_updates = int(raw.get("learning_model_updates", self.learning_model_updates))
            self.learning_brier_sum = float(raw.get("learning_brier_sum", self.learning_brier_sum))
            self.baseline_brier_sum = float(raw.get("baseline_brier_sum", self.baseline_brier_sum))
            self.learning_baseline_brier_sum = float(
                raw.get("learning_baseline_brier_sum", self.learning_baseline_brier_sum)
            )
            recent_lift = raw.get("recent_brier_lift", [])
            if isinstance(recent_lift, list):
                for v in recent_lift[-RECENT_LIFT_WINDOW:]:
                    try:
                        self._recent_brier_lift.append(float(v))
                    except Exception:
                        continue
            recent_learning_lift = raw.get("recent_learning_brier_lift", [])
            if isinstance(recent_learning_lift, list):
                for v in recent_learning_lift[-RECENT_LIFT_WINDOW:]:
                    try:
                        self._recent_learning_brier_lift.append(float(v))
                    except Exception:
                        continue
            settled_history = raw.get("settled_history", [])
            if isinstance(settled_history, list):
                for item in settled_history[-500:]:
                    if isinstance(item, dict):
                        self._settled_history.append(item)
            prediction_history = raw.get("prediction_history", [])
            if isinstance(prediction_history, list):
                for v in prediction_history[-self._prediction_history.maxlen:]:
                    try:
                        self._prediction_history.append(float(v))
                    except Exception:
                        continue
            feature_drift_stats = raw.get("feature_drift_stats", {})
            if isinstance(feature_drift_stats, dict):
                self._feature_drift_stats = {
                    str(k): dict(v)
                    for k, v in feature_drift_stats.items()
                    if isinstance(v, dict)
                }
            update_rejections = raw.get("update_rejections", [])
            if isinstance(update_rejections, list):
                for item in update_rejections[-200:]:
                    if isinstance(item, dict):
                        self._update_rejections.append(item)
            self._consecutive_gate_passes = int(raw.get("consecutive_gate_passes", self._consecutive_gate_passes))
            last_gate_state = raw.get("last_gate_state", self._last_gate_state)
            self._last_gate_state = last_gate_state if isinstance(last_gate_state, bool) else None
            self.stake_fraction = float(raw.get("stake_fraction", self.stake_fraction))
            self.min_edge = float(raw.get("min_edge", self.min_edge))
            pending = raw.get("pending", [])
            if isinstance(pending, list):
                for p in pending:
                    try:
                        bet = _PendingBet(
                            bet_id=str(p["bet_id"]),
                            market_id=str(p["market_id"]),
                            selection_id=str(p["selection_id"]),
                            selection_name=str(p.get("selection_name", "")),
                            event_name=str(p.get("event_name", "")),
                            entry_odds=float(p["entry_odds"]),
                            stake=float(p["stake"]),
                            base_prob=float(p["base_prob"]),
                            predicted_prob=float(p["predicted_prob"]),
                            features={k: float(v) for k, v in dict(p.get("features", {})).items()},
                        )
                        self._pending[(bet.market_id, bet.selection_id)] = bet
                    except Exception:
                        continue
            learning_candidates = raw.get("learning_candidates", [])
            if isinstance(learning_candidates, list):
                for c in learning_candidates:
                    try:
                        cand = _LearningCandidate(
                            market_id=str(c["market_id"]),
                            selection_id=str(c["selection_id"]),
                            selection_name=str(c.get("selection_name", "")),
                            event_name=str(c.get("event_name", "")),
                            entry_odds=float(c["entry_odds"]),
                            base_prob=float(c["base_prob"]),
                            predicted_prob=float(c["predicted_prob"]),
                            features={k: float(v) for k, v in dict(c.get("features", {})).items()},
                        )
                        self._learning_candidates[cand.market_id] = cand
                    except Exception:
                        continue
        except Exception:
            pass

    def _features_from_snapshot(
        self,
        snapshot: PriceSnapshot,
        market_start: Optional[datetime],
    ) -> Dict[str, float]:
        prev = self._prev_snapshots.get(snapshot.market_id)
        micro = build_market_microstructure(snapshot, market_start=market_start, previous_snapshot=prev)
        return {
            "spread_mean": float(micro.spread_mean),
            "imbalance": float(micro.imbalance),
            "depth_total_eur": float(micro.depth_total_eur),
            "price_velocity": float(micro.price_velocity),
            "short_volatility": float(micro.short_volatility),
            "time_to_start_sec": float(micro.time_to_start_sec),
            "in_play": 1.0 if micro.in_play else 0.0,
            "weighted_spread": float(micro.weighted_spread),
            "lay_back_ratio": float(micro.lay_back_ratio),
            "top_of_book_concentration": float(micro.top_of_book_concentration),
            "selection_count": float(micro.selection_count),
            "volume_momentum": float(micro.volume_momentum),
            "back_lay_crossover": float(micro.back_lay_crossover),
            "overround_distance": float(micro.overround_distance),
            "depth_ratio_top3": float(micro.depth_ratio_top3),
            "price_range": float(micro.price_range),
        }

    def _predict_prob(self, base_prob: float, features: Dict[str, float]) -> float:
        if self.model_kind == "implied_market":
            return max(1e-6, min(1.0 - 1e-6, base_prob))
        if self.model_kind == "residual_logit":
            return self.model.predict_proba(base_prob, features)
        return self.model.predict_proba(features)

    def _append_example(self, ex: PredictionExample) -> None:
        payload = {
            "timestamp": ex.timestamp,
            "base_prob": ex.base_prob,
            "odds": ex.odds,
            "label": ex.label,
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            **ex.features,
        }
        with self._examples_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _is_control_model(self) -> bool:
        return self.model_kind == "implied_market"

    def _record_prediction(self, prob: float) -> None:
        if math.isfinite(prob):
            self._prediction_history.append(float(prob))

    def _prediction_is_frozen(self) -> bool:
        w = max(5, int(config.PREDICTION_FROZEN_WINDOW))
        if len(self._prediction_history) < w:
            return False
        vals = list(self._prediction_history)[-w:]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        return math.sqrt(var) <= float(config.PREDICTION_FROZEN_STD_THRESHOLD)

    def _prediction_saturation_rate(self) -> float:
        w = max(10, int(config.PREDICTION_SATURATION_WINDOW))
        if len(self._prediction_history) < w:
            return 0.0
        vals = list(self._prediction_history)[-w:]
        low = float(config.PREDICTION_SATURATION_LOW)
        high = float(config.PREDICTION_SATURATION_HIGH)
        sat = sum(1 for p in vals if p <= low or p >= high)
        return sat / len(vals)

    def _update_feature_drift(self, features: Dict[str, float], market_id: str) -> List[dict]:
        events: List[dict] = []
        min_count = max(2, int(config.PREDICTION_DRIFT_MIN_COUNT))
        z_thresh = float(config.PREDICTION_DRIFT_Z_THRESHOLD)
        sustain = max(1, int(config.PREDICTION_DRIFT_SUSTAIN_COUNT))
        for name, raw in features.items():
            x = float(raw)
            if not math.isfinite(x):
                continue
            st = self._feature_drift_stats.setdefault(name, {"n": 0.0, "mean": 0.0, "m2": 0.0, "streak": 0.0})
            n = int(st.get("n", 0))
            mean = float(st.get("mean", 0.0))
            m2 = float(st.get("m2", 0.0))
            if n >= min_count:
                var = m2 / max(1, n - 1)
                sd = math.sqrt(max(var, 1e-12))
                z = abs(x - mean) / sd if sd > 0 else 0.0
                if z >= z_thresh:
                    st["streak"] = float(st.get("streak", 0.0)) + 1.0
                else:
                    st["streak"] = 0.0
                if int(st["streak"]) == sustain:
                    events.append(
                        {
                            "kind": "prediction_drift_alert",
                            "model_id": self.model_id,
                            "market_id": market_id,
                            "feature": name,
                            "z_score": round(z, 4),
                            "value": round(x, 6),
                        }
                    )
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
            st["n"] = float(n)
            st["mean"] = float(mean)
            st["m2"] = float(m2)
        return events

    def _validate_features(self, features: Dict[str, float], market_id: str, context: str) -> Optional[dict]:
        abs_max = float(config.PREDICTION_FEATURE_ABS_MAX)
        for name, v in features.items():
            x = float(v)
            if not math.isfinite(x):
                ev = {
                    "kind": "prediction_update_rejected",
                    "model_id": self.model_id,
                    "market_id": market_id,
                    "context": context,
                    "reason": "non_finite_feature",
                    "feature": name,
                }
                self._update_rejections.append(ev)
                return ev
            if abs(x) > abs_max:
                ev = {
                    "kind": "prediction_update_rejected",
                    "model_id": self.model_id,
                    "market_id": market_id,
                    "context": context,
                    "reason": "feature_abs_limit",
                    "feature": name,
                    "value": round(x, 4),
                    "limit": abs_max,
                }
                self._update_rejections.append(ev)
                return ev
        return None

    def _rolling_metrics(self, window: int) -> Dict[str, object]:
        hist = list(self._settled_history)[-window:]
        settled = len(hist)
        if settled <= 0:
            return {
                "settled": 0,
                "model_brier": 0.0,
                "baseline_brier": 0.0,
                "brier_lift_abs": 0.0,
                "roi_pct": 0.0,
                "clv_avg": None,
            }
        model_brier = sum(float(h.get("model_brier", 0.0)) for h in hist) / settled
        baseline_brier = sum(float(h.get("baseline_brier", 0.0)) for h in hist) / settled
        pnl = sum(float(h.get("pnl", 0.0)) for h in hist)
        stake = sum(max(0.0, float(h.get("stake", 0.0))) for h in hist)
        roi_pct = (pnl / stake * 100.0) if stake > 0 else 0.0
        clv_vals = [float(h["clv"]) for h in hist if isinstance(h.get("clv"), (int, float))]
        clv_avg = (sum(clv_vals) / len(clv_vals)) if clv_vals else None
        return {
            "settled": settled,
            "model_brier": round(model_brier, 6),
            "baseline_brier": round(baseline_brier, 6),
            "brier_lift_abs": round(baseline_brier - model_brier, 6),
            "roi_pct": round(roi_pct, 4),
            "clv_avg": round(clv_avg, 6) if clv_avg is not None else None,
        }

    def _strict_gate_status(self) -> Tuple[bool, str, Dict[str, object], Dict[str, object], Dict[str, object]]:
        r50 = self._rolling_metrics(50)
        r100 = self._rolling_metrics(100)
        r200 = self._rolling_metrics(200)
        if self._is_control_model():
            return False, "control_model", r50, r100, r200
        if int(self.settled_bets) < int(config.PREDICTION_STRICT_GATE_MIN_SETTLED):
            return False, "insufficient_settled_bets", r50, r100, r200
        if int(r100["settled"]) < 100 or int(r200["settled"]) < 200:
            return False, "insufficient_window_coverage", r50, r100, r200
        if float(r100["brier_lift_abs"]) <= 0 or float(r200["brier_lift_abs"]) <= 0:
            return False, "negative_brier_lift", r50, r100, r200
        if float(r100["roi_pct"]) < 0 or float(r200["roi_pct"]) < 0:
            return False, "negative_roi", r50, r100, r200
        return True, "pass", r50, r100, r200

    def _effective_policy(self) -> Tuple[float, float]:
        gate_pass, _, _, _, _ = self._strict_gate_status()
        mode = self._enforcement_mode
        if mode not in {"soft", "strict"}:
            return 1.0, self.min_edge
        if gate_pass:
            if self._consecutive_gate_passes >= self._pass_streak_required:
                return 1.0, self.min_edge
            # keep conservative until pass streak is met
            return 0.5, self.min_edge
        fail_mult = float(config.PREDICTION_GATE_FAIL_STAKE_MULTIPLIER)
        if mode == "soft":
            fail_mult = max(0.5, fail_mult)
        return fail_mult, self.min_edge + float(config.PREDICTION_GATE_FAIL_MIN_EDGE_BUMP)

    def _git_sha(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except Exception:
            return "unknown"

    def _config_hash(self) -> str:
        payload = {
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "stake_fraction": self.stake_fraction,
            "min_edge": self.min_edge,
            "min_liquidity": self.min_liquidity,
            "enforcement": self._enforcement_mode,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def _maybe_log_experiment_snapshot(self) -> None:
        every = max(1, int(config.PREDICTION_EXPERIMENT_LOG_EVERY_SETTLED))
        if self.settled_bets <= 0 or (self.settled_bets % every) != 0:
            return
        gate_pass, gate_reason, r50, r100, r200 = self._strict_gate_status()
        self._experiment_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": f"{self.model_id}-{self.settled_bets}",
            "ts_start": None,
            "ts_end": datetime.now(timezone.utc).isoformat(),
            "git_sha": self._git_sha(),
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "config_hash": self._config_hash(),
            "models": {
                "model_path": str(self.model_path),
                "model_updates": self.update_count,
            },
            "metrics": {"rolling_50": r50, "rolling_100": r100, "rolling_200": r200},
            "gate": {"strict_gate_pass": gate_pass, "reason": gate_reason},
        }
        with self._experiment_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _update_model(self, ex: PredictionExample) -> None:
        if self.model_kind == "implied_market":
            return
        self.model.fit([ex], epochs=1, lr=0.03)
        self.update_count += 1
        self._maybe_save_model()
        self._save_state()

    def _stake_size(self, stake_multiplier: float = 1.0) -> float:
        stake = self.balance * self.stake_fraction * max(0.0, float(stake_multiplier))
        stake = max(self.min_stake, stake)
        stake = min(self.max_stake, stake, self.balance)
        return round(stake, 2)

    def _reset_if_bust(self) -> bool:
        if self.balance > 0:
            return False
        self.resets += 1
        self.balance = self.initial_balance
        self._pending.clear()
        self._save_state()
        return True

    def _select_learning_candidate(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        features: Dict[str, float],
    ) -> Optional[_LearningCandidate]:
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") == "CLOSED":
            return None
        best = None
        for sel in snapshot.selections:
            odds = float(sel.best_back_price)
            liq = float(sel.available_to_back)
            if odds <= 1.01 or liq < self.min_liquidity:
                continue
            base_prob = 1.0 / odds
            pred_prob = self._predict_prob(base_prob, features)
            self._record_prediction(pred_prob)
            if best is None or liq > best["liq"]:
                best = {
                    "selection_id": sel.selection_id,
                    "selection_name": sel.name,
                    "entry_odds": odds,
                    "base_prob": base_prob,
                    "predicted_prob": pred_prob,
                    "liq": liq,
                }
        if best is None:
            return None
        return _LearningCandidate(
            market_id=market_id,
            selection_id=best["selection_id"],
            selection_name=best["selection_name"],
            event_name=event_name,
            entry_odds=best["entry_odds"],
            base_prob=best["base_prob"],
            predicted_prob=best["predicted_prob"],
            features=features,
        )

    def _maybe_track_learning_candidate(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        features: Dict[str, float],
    ) -> Optional[dict]:
        if market_id in self._learning_candidates:
            return None
        if any(b.market_id == market_id for b in self._pending.values()):
            return None
        cand = self._select_learning_candidate(market_id, snapshot, event_name, features)
        if cand is None:
            return None
        self._learning_candidates[market_id] = cand
        self.learning_tracked += 1
        self._save_state()
        return {
            "kind": "prediction_learning_track",
            "model_id": self.model_id,
            "market_id": market_id,
            "selection": cand.selection_name,
            "pred_prob": round(cand.predicted_prob, 5),
            "base_prob": round(cand.base_prob, 5),
        }

    def _settle_learning_candidate(self, snapshot: PriceSnapshot) -> List[dict]:
        events: List[dict] = []
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") != "CLOSED":
            return events
        cand = self._learning_candidates.get(snapshot.market_id)
        if cand is None:
            return events
        by_id = {s.selection_id: s for s in snapshot.selections}
        sel = by_id.get(cand.selection_id)
        if sel is None:
            return events
        status = str(getattr(sel, "runner_status", "UNKNOWN") or "UNKNOWN").upper()
        if status == "WINNER":
            y = 1
        elif status == "LOSER":
            y = 0
        else:
            # Removed/void/unknown labels are not useful for supervised updates.
            if status in {"REMOVED", "REMOVED_VACANT", "VOID", "HIDDEN"}:
                self._learning_candidates.pop(snapshot.market_id, None)
                self._save_state()
            return events

        ex = PredictionExample(
            timestamp=datetime.now(timezone.utc).isoformat(),
            base_prob=cand.base_prob,
            odds=cand.entry_odds,
            label=y,
            features=cand.features,
        )
        rejection = self._validate_features(cand.features, cand.market_id, context="learning")
        if rejection is not None:
            self._learning_candidates.pop(snapshot.market_id, None)
            self._save_state()
            events.append(rejection)
            return events
        self.learning_settled += 1
        model_brier = (cand.predicted_prob - y) ** 2
        baseline_brier = (cand.base_prob - y) ** 2
        self.learning_brier_sum += model_brier
        self.learning_baseline_brier_sum += baseline_brier
        self._recent_learning_brier_lift.append(baseline_brier - model_brier)
        if self.model_kind != "implied_market":
            self._update_model(ex)
            self.learning_model_updates += 1
        self._append_example(ex)
        self._learning_candidates.pop(snapshot.market_id, None)
        self._save_state()
        events.append(
            {
                "kind": "prediction_learning_settle",
                "model_id": self.model_id,
                "market_id": cand.market_id,
                "selection": cand.selection_name,
                "won": y == 1,
                "updated": self.model_kind != "implied_market",
                "pred_prob": round(cand.predicted_prob, 5),
                "base_prob": round(cand.base_prob, 5),
                "brier_lift": round(baseline_brier - model_brier, 6),
            }
        )
        return events

    def _settle_pending(self, snapshot: PriceSnapshot) -> List[dict]:
        events: List[dict] = []
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") != "CLOSED":
            return events
        to_remove = []
        by_id = {s.selection_id: s for s in snapshot.selections}

        for key, bet in list(self._pending.items()):
            if bet.market_id != snapshot.market_id:
                continue
            sel = by_id.get(bet.selection_id)
            if sel is None:
                continue
            status = str(getattr(sel, "runner_status", "UNKNOWN") or "UNKNOWN").upper()
            if status == "WINNER":
                y = 1
            elif status == "LOSER":
                y = 0
            elif status in {"REMOVED", "REMOVED_VACANT", "VOID", "HIDDEN"}:
                self.voids += 1
                self.balance += bet.stake
                events.append(
                    {
                        "kind": "prediction_settle",
                        "model_id": self.model_id,
                        "market_id": bet.market_id,
                        "selection": bet.selection_name,
                        "won": False,
                        "void": True,
                        "pnl_eur": 0.0,
                        "balance_eur": round(self.balance, 2),
                        "reset": False,
                    }
                )
                to_remove.append(key)
                self._save_state()
                continue
            else:
                continue
            ex = PredictionExample(
                timestamp=datetime.now(timezone.utc).isoformat(),
                base_prob=bet.base_prob,
                odds=bet.entry_odds,
                label=y,
                features=bet.features,
            )
            rejection = self._validate_features(bet.features, bet.market_id, context="settlement")
            if rejection is not None:
                to_remove.append(key)
                events.append(rejection)
                self._save_state()
                continue
            self._update_model(ex)
            self._append_example(ex)

            model_brier = (bet.predicted_prob - y) ** 2
            baseline_brier = (bet.base_prob - y) ** 2
            self.brier_sum += model_brier
            self.baseline_brier_sum += baseline_brier
            self._recent_brier_lift.append(baseline_brier - model_brier)
            self.settled_bets += 1
            if y == 1:
                self.wins += 1
                payout = bet.stake * bet.entry_odds
                pnl = payout - bet.stake
                self.balance += payout
            else:
                self.losses += 1
                pnl = -bet.stake
            self.total_pnl += pnl
            clv_value = self._clv_tracker.compute_clv(bet.bet_id) if self._clv_tracker else None
            self._settled_history.append(
                {
                    "market_id": bet.market_id,
                    "selection_id": bet.selection_id,
                    "stake": float(bet.stake),
                    "pnl": float(pnl),
                    "model_brier": float(model_brier),
                    "baseline_brier": float(baseline_brier),
                    "clv": clv_value if isinstance(clv_value, (int, float)) else None,
                    "pred_prob": float(bet.predicted_prob),
                    "base_prob": float(bet.base_prob),
                }
            )
            busted = self._reset_if_bust()
            events.append(
                {
                    "kind": "prediction_settle",
                    "model_id": self.model_id,
                    "market_id": bet.market_id,
                    "selection": bet.selection_name,
                    "won": y == 1,
                    "void": False,
                    "pnl_eur": round(pnl, 4),
                    "balance_eur": round(self.balance, 2),
                    "reset": busted,
                    "brier_lift": round(baseline_brier - model_brier, 6),
                    "clv": clv_value,
                }
            )
            gate_pass, gate_reason, _, _, _ = self._strict_gate_status()
            if gate_pass:
                self._consecutive_gate_passes += 1
            else:
                self._consecutive_gate_passes = 0
            if self._last_gate_state is None or gate_pass != self._last_gate_state:
                events.append(
                    {
                        "kind": "prediction_gate_pass" if gate_pass else "prediction_gate_fail",
                        "model_id": self.model_id,
                        "market_id": bet.market_id,
                        "reason": gate_reason,
                        "settled_bets": self.settled_bets,
                    }
                )
            self._last_gate_state = gate_pass
            self._maybe_log_experiment_snapshot()
            to_remove.append(key)
            self._save_state()

        for key in to_remove:
            self._pending.pop(key, None)
        if to_remove:
            self._save_state()
        return events

    def _open_bet(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        features: Dict[str, float],
    ) -> Optional[dict]:
        if str(getattr(snapshot, "market_status", "OPEN") or "OPEN") == "CLOSED":
            return None
        stake_multiplier, effective_min_edge = self._effective_policy()
        if self._prediction_is_frozen():
            return None
        if self._prediction_saturation_rate() >= float(config.PREDICTION_SATURATION_RATE_THRESHOLD):
            return None
        best = None
        for sel in snapshot.selections:
            odds = float(sel.best_back_price)
            liq = float(sel.available_to_back)
            if odds <= 1.01 or liq < self.min_liquidity:
                continue
            base_prob = 1.0 / odds
            pred_prob = self._predict_prob(base_prob, features)
            self._record_prediction(pred_prob)
            edge = pred_prob - base_prob
            if best is None or edge > best["edge"]:
                best = {
                    "selection_id": sel.selection_id,
                    "selection_name": sel.name,
                    "odds": odds,
                    "base_prob": base_prob,
                    "pred_prob": pred_prob,
                    "edge": edge,
                }

        if best is None or best["edge"] < effective_min_edge:
            return None

        policy_gate = get_model_policy(self.model_id)
        policy_mode = str(policy_gate.get("mode", "execute") or "execute").lower()
        if policy_mode != "execute":
            return None
        policy_stake_multiplier = float(policy_gate.get("stake_multiplier", 1.0) or 1.0)
        if policy_stake_multiplier <= 0:
            return None

        key = (market_id, best["selection_id"])
        if key in self._pending:
            return None

        stake = self._stake_size(stake_multiplier=(stake_multiplier * policy_stake_multiplier))
        if stake < self.min_stake or stake <= 0:
            return None

        self.balance -= stake
        self.total_bets += 1
        self.last_edge = best["edge"]
        self.last_prediction = best["pred_prob"]

        bet_id = f"{self.model_id}:{market_id}:{best['selection_id']}:{int(datetime.now(timezone.utc).timestamp())}"
        # If we had a shadow learning candidate for this market, remove it to avoid
        # double-labeling the same market for the same model.
        self._learning_candidates.pop(market_id, None)
        self._pending[key] = _PendingBet(
            bet_id=bet_id,
            market_id=market_id,
            selection_id=best["selection_id"],
            selection_name=best["selection_name"],
            event_name=event_name,
            entry_odds=best["odds"],
            stake=stake,
            base_prob=best["base_prob"],
            predicted_prob=best["pred_prob"],
            features=features,
        )
        if self._clv_tracker is not None:
            self._clv_tracker.record_entry(
                bet_id=bet_id,
                market_id=market_id,
                selection_id=best["selection_id"],
                entry_odds=best["odds"],
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
            )
        self._save_state()
        return {
            "kind": "prediction_open",
            "model_id": self.model_id,
            "market_id": market_id,
            "event_name": event_name,
            "selection": best["selection_name"],
            "odds": round(best["odds"], 3),
            "edge": round(best["edge"], 5),
            "pred_prob": round(best["pred_prob"], 5),
            "stake_eur": stake,
            "stake_multiplier": round(stake_multiplier * policy_stake_multiplier, 4),
            "effective_min_edge": round(effective_min_edge, 6),
            "balance_eur": round(self.balance, 2),
            "bet_id": bet_id,
            "policy_gate_mode": policy_mode,
            "policy_gate_reason": str(policy_gate.get("reason", "")),
        }

    def process_snapshot(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        market_start: Optional[datetime],
    ) -> Dict[str, object]:
        events: List[dict] = []
        if self._clv_tracker is not None:
            self._clv_tracker.record_closing_prices(market_id, snapshot)
        settle_events = self._settle_pending(snapshot)
        learning_events = self._settle_learning_candidate(snapshot)
        events.extend(settle_events)
        events.extend(learning_events)
        features = self._features_from_snapshot(snapshot, market_start)
        events.extend(self._update_feature_drift(features, market_id=market_id))
        if not any(e.get("reset") for e in settle_events):
            opened = self._open_bet(market_id, snapshot, event_name, features)
            if opened:
                events.append(opened)
            else:
                tracked = self._maybe_track_learning_candidate(
                    market_id=market_id,
                    snapshot=snapshot,
                    event_name=event_name,
                    features=features,
                )
                if tracked:
                    events.append(tracked)
        self._prev_snapshots[market_id] = snapshot
        return {"events": events, "state": self.get_state()}

    def process_settlement_snapshot(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
    ) -> Dict[str, object]:
        """
        Settle existing pending bets only, without opening new positions.
        Used for markets that dropped out of scan universe but still have open bets.
        """
        events: List[dict] = []
        if self._clv_tracker is not None:
            self._clv_tracker.record_closing_prices(market_id, snapshot)
        settle_events = self._settle_pending(snapshot)
        learning_events = self._settle_learning_candidate(snapshot)
        events.extend(settle_events)
        events.extend(learning_events)
        self._prev_snapshots[market_id] = snapshot
        return {"events": events, "state": self.get_state()}

    def get_state(self) -> Dict[str, object]:
        avg_brier = (self.brier_sum / self.settled_bets) if self.settled_bets > 0 else 0.0
        baseline_avg_brier = (
            self.baseline_brier_sum / self.settled_bets if self.settled_bets > 0 else 0.0
        )
        learning_avg_brier = (
            self.learning_brier_sum / self.learning_settled if self.learning_settled > 0 else 0.0
        )
        learning_baseline_avg_brier = (
            self.learning_baseline_brier_sum / self.learning_settled if self.learning_settled > 0 else 0.0
        )
        brier_lift = baseline_avg_brier - avg_brier
        learning_brier_lift = learning_baseline_avg_brier - learning_avg_brier
        win_rate = (self.wins / self.settled_bets) if self.settled_bets > 0 else 0.0
        roi = (self.total_pnl / self.initial_balance) if self.initial_balance > 0 else 0.0
        recent_brier_lift = (
            sum(self._recent_brier_lift) / len(self._recent_brier_lift)
            if len(self._recent_brier_lift) > 0 else 0.0
        )
        recent_learning_brier_lift = (
            sum(self._recent_learning_brier_lift) / len(self._recent_learning_brier_lift)
            if len(self._recent_learning_brier_lift) > 0 else 0.0
        )
        gate_pass, gate_reason, r50, r100, r200 = self._strict_gate_status()
        saturation_rate = self._prediction_saturation_rate()
        frozen = self._prediction_is_frozen()
        policy_gate = get_model_policy(self.model_id)
        return {
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "enabled": True,
            "stake_fraction": self.stake_fraction,
            "min_edge": self.min_edge,
            "balance_eur": round(self.balance, 2),
            "initial_balance_eur": round(self.initial_balance, 2),
            "total_bets": self.total_bets,
            "settled_bets": self.settled_bets,
            "wins": self.wins,
            "losses": self.losses,
            "voids": self.voids,
            "win_rate": round(win_rate, 4),
            "roi_pct": round(roi * 100, 4),
            "total_pnl_eur": round(self.total_pnl, 4),
            "resets": self.resets,
            "open_positions": len(self._pending),
            "avg_brier": round(avg_brier, 6),
            "baseline_avg_brier": round(baseline_avg_brier, 6),
            "brier_lift_abs": round(brier_lift, 6),
            "model_updates": self.update_count,
            "learning_tracked": self.learning_tracked,
            "learning_settled": self.learning_settled,
            "learning_updates": self.learning_model_updates,
            "learning_avg_brier": round(learning_avg_brier, 6),
            "learning_baseline_avg_brier": round(learning_baseline_avg_brier, 6),
            "learning_brier_lift_abs": round(learning_brier_lift, 6),
            "learning_open_markets": len(self._learning_candidates),
            "recent_brier_lift": round(recent_brier_lift, 6),
            "recent_learning_brier_lift": round(recent_learning_brier_lift, 6),
            "rolling_50": r50,
            "rolling_100": r100,
            "rolling_200": r200,
            "strict_gate_pass": gate_pass,
            "strict_gate_reason": gate_reason,
            "gate_pass_streak": int(self._consecutive_gate_passes),
            "prediction_frozen": frozen,
            "prediction_saturation_rate": round(saturation_rate, 6),
            "update_rejection_count": len(self._update_rejections),
            "model_path": str(self.model_path),
            "model_version": f"{self.model_kind}_online_v1",
            "last_edge": round(self.last_edge, 6),
            "last_prediction": round(self.last_prediction, 6),
            "policy_gate_mode": str(policy_gate.get("mode", "execute") or "execute"),
            "policy_gate_reason": str(policy_gate.get("reason", "") or ""),
            "policy_gate_test_roi": float(policy_gate.get("roi", 0.0) or 0.0),
            "policy_gate_brier_lift": float(policy_gate.get("brier_lift", 0.0) or 0.0),
            "policy_gate_test_bets": int(policy_gate.get("bets", 0) or 0),
        }


class MultiModelPredictionManager:
    """Runs multiple prediction engines in parallel and aggregates telemetry."""

    def __init__(
        self,
        model_kinds: List[str],
        initial_balance_eur: float,
        stake_fraction: float,
        min_stake_eur: float,
        max_stake_eur: float,
        min_edge: float,
        min_liquidity_eur: float,
        model_dir: str,
        save_every: int = 25,
        clv_tracker: Optional[CLVTracker] = None,
        state_dir: Optional[str] = None,
    ):
        self.engines: Dict[str, OnlinePredictionEngine] = {}
        model_root = Path(model_dir)
        model_root.mkdir(parents=True, exist_ok=True)
        state_root = Path(state_dir) if state_dir else None
        if state_root is not None:
            state_root.mkdir(parents=True, exist_ok=True)
        for idx, kind in enumerate(model_kinds):
            model_id = f"{kind}_{idx+1}"
            model_path = str(model_root / f"{model_id}.json")
            model_state_path = str(state_root / f"{model_id}.json") if state_root is not None else None
            self.engines[model_id] = OnlinePredictionEngine(
                model_id=model_id,
                model_kind=kind,
                initial_balance_eur=initial_balance_eur,
                stake_fraction=stake_fraction,
                min_stake_eur=min_stake_eur,
                max_stake_eur=max_stake_eur,
                min_edge=min_edge,
                min_liquidity_eur=min_liquidity_eur,
                model_path=model_path,
                save_every=save_every,
                clv_tracker=clv_tracker,
                state_path=model_state_path,
            )

    def process_snapshot(
        self,
        market_id: str,
        snapshot: PriceSnapshot,
        event_name: str,
        market_start: Optional[datetime],
    ) -> Dict[str, object]:
        events: List[dict] = []
        models: Dict[str, dict] = {}
        for model_id, engine in self.engines.items():
            out = engine.process_snapshot(market_id, snapshot, event_name, market_start)
            events.extend(out.get("events", []))
            models[model_id] = out.get("state", {})
        return {"events": events, "models": models}

    def initial_state(self) -> Dict[str, dict]:
        return {model_id: engine.get_state() for model_id, engine in self.engines.items()}

    def pending_market_ids(self) -> List[str]:
        pending: set[str] = set()
        for engine in self.engines.values():
            pending.update(engine.pending_market_ids())
        return sorted(pending)

    def process_pending_settlements(
        self,
        market_ids: List[str],
        price_cache,
    ) -> Dict[str, object]:
        """
        Run settlement-only updates for the provided market IDs.
        This is intentionally separate from process_snapshot to avoid opening
        additional bets on out-of-universe markets.
        """
        events: List[dict] = []
        models: Dict[str, dict] = {}
        for market_id in market_ids:
            snapshot = price_cache.get_prices(market_id)
            if snapshot is None:
                continue
            for model_id, engine in self.engines.items():
                if not engine.has_pending_market(market_id):
                    continue
                out = engine.process_settlement_snapshot(market_id, snapshot)
                events.extend(out.get("events", []))
                models[model_id] = out.get("state", {})
        if not models:
            models = self.initial_state()
        return {"events": events, "models": models}
