"""
Shared learning-quality tracker for funding models.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import config

ROLLING_WINDOWS = (50, 100, 200)


class FundingLearningQuality:
    """Tracks settled outcomes, strict gate status, and quality sentinels."""

    def __init__(self, model_id: str, model_family: str, state_path: str):
        self.model_id = model_id
        self.model_family = model_family
        self._state_path = Path(state_path)
        self._experiment_log_path = Path(config.FUNDING_EXPERIMENT_LOG_PATH)
        self._settled_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._prediction_history: Deque[float] = deque(
            maxlen=max(300, int(config.FUNDING_SATURATION_WINDOW))
        )
        self._feature_drift_stats: Dict[str, Dict[str, float]] = {}
        self._update_rejections: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._last_gate_state: Optional[bool] = None
        self._load_state()

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "settled_history": list(self._settled_history),
            "prediction_history": list(self._prediction_history),
            "feature_drift_stats": self._feature_drift_stats,
            "update_rejections": list(self._update_rejections),
            "last_gate_state": self._last_gate_state,
        }
        self._state_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        settled = raw.get("settled_history", [])
        if isinstance(settled, list):
            for item in settled[-self._settled_history.maxlen :]:
                if isinstance(item, dict):
                    self._settled_history.append(item)
        preds = raw.get("prediction_history", [])
        if isinstance(preds, list):
            for p in preds[-self._prediction_history.maxlen :]:
                try:
                    self._prediction_history.append(float(p))
                except Exception:
                    continue
        drift = raw.get("feature_drift_stats", {})
        if isinstance(drift, dict):
            self._feature_drift_stats = {
                str(k): dict(v) for k, v in drift.items() if isinstance(v, dict)
            }
        rejects = raw.get("update_rejections", [])
        if isinstance(rejects, list):
            for item in rejects[-200:]:
                if isinstance(item, dict):
                    self._update_rejections.append(item)
        lgs = raw.get("last_gate_state", None)
        self._last_gate_state = lgs if isinstance(lgs, bool) else None

    def add_prediction(self, prob: Optional[float]) -> None:
        if prob is None:
            return
        try:
            p = float(prob)
        except Exception:
            return
        if math.isfinite(p):
            self._prediction_history.append(p)

    def validate_features(
        self, features: Dict[str, float], symbol: str, context: str
    ) -> Optional[Dict[str, Any]]:
        abs_max = float(config.FUNDING_FEATURE_ABS_MAX)
        for name, raw in features.items():
            x = float(raw)
            if not math.isfinite(x):
                ev = {
                    "kind": "funding_update_rejected",
                    "model_id": self.model_id,
                    "symbol": symbol,
                    "context": context,
                    "reason": "non_finite_feature",
                    "feature": name,
                }
                self._update_rejections.append(ev)
                self._save_state()
                return ev
            if abs(x) > abs_max:
                ev = {
                    "kind": "funding_update_rejected",
                    "model_id": self.model_id,
                    "symbol": symbol,
                    "context": context,
                    "reason": "feature_abs_limit",
                    "feature": name,
                    "value": round(x, 6),
                    "limit": abs_max,
                }
                self._update_rejections.append(ev)
                self._save_state()
                return ev
        return None

    def update_feature_drift(
        self, features: Dict[str, float], symbol: str
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        min_count = max(2, int(config.FUNDING_DRIFT_MIN_COUNT))
        z_thresh = float(config.FUNDING_DRIFT_Z_THRESHOLD)
        sustain = max(1, int(config.FUNDING_DRIFT_SUSTAIN_COUNT))
        for name, raw in features.items():
            x = float(raw)
            if not math.isfinite(x):
                continue
            st = self._feature_drift_stats.setdefault(
                name, {"n": 0.0, "mean": 0.0, "m2": 0.0, "streak": 0.0}
            )
            n = int(st.get("n", 0))
            mean = float(st.get("mean", 0.0))
            m2 = float(st.get("m2", 0.0))
            if n >= min_count:
                var = m2 / max(1, n - 1)
                sd = math.sqrt(max(var, 1e-12))
                z = abs(x - mean) / sd if sd > 0 else 0.0
                st["streak"] = float(st.get("streak", 0.0)) + 1.0 if z >= z_thresh else 0.0
                if int(st["streak"]) == sustain:
                    events.append(
                        {
                            "kind": "funding_drift_alert",
                            "model_id": self.model_id,
                            "symbol": symbol,
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
        if events:
            self._save_state()
        return events

    def prediction_is_frozen(self) -> bool:
        w = max(5, int(config.FUNDING_FROZEN_WINDOW))
        if len(self._prediction_history) < w:
            return False
        vals = list(self._prediction_history)[-w:]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        return math.sqrt(var) <= float(config.FUNDING_FROZEN_STD_THRESHOLD)

    def saturation_rate(self) -> float:
        w = max(10, int(config.FUNDING_SATURATION_WINDOW))
        if len(self._prediction_history) < w:
            return 0.0
        vals = list(self._prediction_history)[-w:]
        low = float(config.FUNDING_SATURATION_LOW)
        high = float(config.FUNDING_SATURATION_HIGH)
        sat = sum(1 for p in vals if p <= low or p >= high)
        return sat / len(vals)

    def record_settlement(
        self,
        symbol: str,
        stake: float,
        pnl: float,
        label: int,
        pred_prob: float,
        base_prob: float,
        clv: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        model_brier = (float(pred_prob) - int(label)) ** 2
        baseline_brier = (float(base_prob) - int(label)) ** 2
        self._settled_history.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "stake": float(stake),
                "pnl": float(pnl),
                "label": int(label),
                "pred_prob": float(pred_prob),
                "base_prob": float(base_prob),
                "model_brier": float(model_brier),
                "baseline_brier": float(baseline_brier),
                "clv": float(clv) if isinstance(clv, (int, float)) else None,
            }
        )
        events: List[Dict[str, Any]] = []
        gate_pass, reason, *_ = self.strict_gate_status()
        if self._last_gate_state is None or gate_pass != self._last_gate_state:
            events.append(
                {
                    "kind": "funding_gate_pass" if gate_pass else "funding_gate_fail",
                    "model_id": self.model_id,
                    "symbol": symbol,
                    "reason": reason,
                }
            )
        self._last_gate_state = gate_pass
        self._maybe_log_experiment()
        self._save_state()
        return events

    def rolling_metrics(self, window: int) -> Dict[str, Any]:
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

    def strict_gate_status(self) -> Tuple[bool, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        r50 = self.rolling_metrics(50)
        r100 = self.rolling_metrics(100)
        r200 = self.rolling_metrics(200)
        if len(self._settled_history) < int(config.FUNDING_STRICT_MIN_SETTLED):
            return False, "insufficient_settled", r50, r100, r200
        if int(r100["settled"]) < 100 or int(r200["settled"]) < 200:
            return False, "insufficient_window_coverage", r50, r100, r200
        if float(r100["brier_lift_abs"]) <= 0 or float(r200["brier_lift_abs"]) <= 0:
            return False, "negative_brier_lift", r50, r100, r200
        if float(r100["roi_pct"]) < 0 or float(r200["roi_pct"]) < 0:
            return False, "negative_roi", r50, r100, r200
        return True, "pass", r50, r100, r200

    def gate_policy(self) -> Tuple[float, float]:
        gate_pass, _, _, _, _ = self.strict_gate_status()
        mode = str(config.FUNDING_GATE_MODE or "observe").lower()
        if mode == "observe":
            return 1.0, 0.0
        if gate_pass:
            return 1.0, 0.0
        if mode == "soft":
            return max(0.0, float(config.FUNDING_GATE_FAIL_SOFT_MULTIPLIER)), float(
                config.FUNDING_GATE_FAIL_EDGE_BUMP
            )
        return max(0.0, float(config.FUNDING_GATE_FAIL_FULL_MULTIPLIER)), float(
            config.FUNDING_GATE_FAIL_EDGE_BUMP
        )

    def state(self) -> Dict[str, Any]:
        gate_pass, gate_reason, r50, r100, r200 = self.strict_gate_status()
        return {
            "rolling_50": r50,
            "rolling_100": r100,
            "rolling_200": r200,
            "strict_gate_pass": gate_pass,
            "strict_gate_reason": gate_reason,
            "eligibility_status": "eligible" if gate_pass else "restricted",
            "prediction_frozen": self.prediction_is_frozen(),
            "prediction_saturation_rate": round(self.saturation_rate(), 6),
            "update_rejections": list(self._update_rejections)[-20:],
            "settled_count": len(self._settled_history),
        }

    def _git_sha(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except Exception:
            return "unknown"

    def _config_hash(self) -> str:
        payload = {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "gate_mode": str(config.FUNDING_GATE_MODE),
            "strict_min_settled": int(config.FUNDING_STRICT_MIN_SETTLED),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def _maybe_log_experiment(self) -> None:
        every = max(1, int(config.FUNDING_EXPERIMENT_LOG_EVERY_SETTLED))
        settled = len(self._settled_history)
        if settled <= 0 or (settled % every) != 0:
            return
        gate_pass, gate_reason, r50, r100, r200 = self.strict_gate_status()
        payload = {
            "run_id": f"{self.model_id}-{settled}",
            "ts_start": None,
            "ts_end": datetime.now(timezone.utc).isoformat(),
            "git_sha": self._git_sha(),
            "config_hash": self._config_hash(),
            "model_family": self.model_family,
            "models": {"model_id": self.model_id},
            "params": {"gate_mode": str(config.FUNDING_GATE_MODE)},
            "metrics": {"rolling_50": r50, "rolling_100": r100, "rolling_200": r200},
            "gate": {"strict_gate_pass": gate_pass, "reason": gate_reason},
            "mode": str(config.FUNDING_GATE_MODE),
        }
        self._experiment_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._experiment_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
