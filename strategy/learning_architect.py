"""
Learning Architect:
- Periodically evaluates model-account performance
- Proposes bounded parameter updates
- Deterministic rules (no local LLM/Ollama dependency)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import config
from monitoring.alerting import alert_model_degradation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchitectDecision:
    ts: str
    mode: str  # "rules"
    applied: bool
    reason: str
    proposals: List[dict]


class LearningArchitect:
    """Meta-controller for online prediction settings."""

    def __init__(self):
        self.interval_seconds = config.ARCHITECT_INTERVAL_SECONDS
        self.min_settled_bets = config.ARCHITECT_MIN_SETTLED_BETS
        self.last_run_ts: Optional[datetime] = None
        self.last_decision: Optional[ArchitectDecision] = None
        self._log_dir = Path(config.ARCHITECT_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "decisions.jsonl"
        self._cooldown_multiplier = 1

    def _should_run(self, now: datetime) -> bool:
        if self.last_run_ts is None:
            return True
        return (now - self.last_run_ts).total_seconds() >= (self.interval_seconds * self._cooldown_multiplier)

    def _clip(self, value: Decimal, low: Decimal, high: Decimal) -> Decimal:
        if value < low:
            return low
        if value > high:
            return high
        return value

    def _log(self, decision: ArchitectDecision) -> None:
        payload = {
            "ts": decision.ts,
            "mode": decision.mode,
            "applied": decision.applied,
            "reason": decision.reason,
            "proposals": decision.proposals,
        }
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _rules_proposals(self, model_states: Dict[str, dict]) -> List[dict]:
        proposals: List[dict] = []
        for model_id, st in model_states.items():
            settled = int(st.get("settled_bets", 0))
            if settled < self.min_settled_bets:
                continue
            roi = Decimal(str(st.get("roi_pct", 0))) / Decimal("100")
            brier = Decimal(str(st.get("avg_brier", 0)))

            stake_fraction = Decimal(str(st.get("stake_fraction", 0.05)))
            min_edge = Decimal(str(st.get("min_edge", 0.03)))

            new_stake = stake_fraction
            new_edge = min_edge
            reason = "stable"
            # Simple bounded policy.
            if roi < Decimal("-0.05"):
                new_stake = config.ARCHITECT_MIN_STAKE_FRACTION
                new_edge = max(min_edge, Decimal("0.02"))
                reason = "forced_min_stake_bad_roi"
            elif roi < Decimal("-0.01") or brier > Decimal("0.30"):
                new_stake = stake_fraction - config.ARCHITECT_MAX_STAKE_STEP
                new_edge = min_edge + config.ARCHITECT_MAX_EDGE_STEP
                reason = "underperforming_risk_down"
            elif roi > Decimal("0.01") and brier < Decimal("0.24"):
                new_stake = stake_fraction + (config.ARCHITECT_MAX_STAKE_STEP / Decimal("2"))
                new_edge = min_edge - (config.ARCHITECT_MAX_EDGE_STEP / Decimal("2"))
                reason = "performing_scale_up"

            new_stake = self._clip(new_stake, config.ARCHITECT_MIN_STAKE_FRACTION, config.ARCHITECT_MAX_STAKE_FRACTION)
            hard_edge_floor = max(config.ARCHITECT_MIN_EDGE, Decimal("0.02"))
            new_edge = self._clip(new_edge, hard_edge_floor, config.ARCHITECT_MAX_EDGE)
            if new_stake != stake_fraction or new_edge != min_edge:
                proposals.append(
                    {
                        "model_id": model_id,
                        "stake_fraction": float(new_stake),
                        "min_edge": float(new_edge),
                        "reason": reason,
                    }
                )
        return proposals

    def evaluate_and_apply(self, prediction_manager) -> ArchitectDecision:
        now = datetime.now(timezone.utc)
        if not self._should_run(now):
            return self.last_decision or ArchitectDecision(
                ts=now.isoformat(),
                mode="rules",
                applied=False,
                reason="interval_not_reached",
                proposals=[],
            )

        states = prediction_manager.initial_state()
        # Degradation alerts
        for model_id, st in states.items():
            settled = int(st.get("settled_bets", 0))
            if settled <= 50:
                continue
            brier = float(st.get("avg_brier", 0.0))
            roi = float(st.get("roi_pct", 0.0))
            win_rate = float(st.get("win_rate", 0.0)) * 100.0
            if brier > 0.30:
                alert_model_degradation(model_id, "brier", brier, 0.30)
            if roi < -10.0:
                alert_model_degradation(model_id, "roi_pct", roi, -10.0)
            if win_rate < 35.0:
                alert_model_degradation(model_id, "win_rate_pct", win_rate, 35.0)

        proposals = self._rules_proposals(states)
        mode = "rules"

        applied = False
        if proposals:
            for p in proposals:
                model_id = p.get("model_id")
                eng = prediction_manager.engines.get(model_id)
                if eng is None:
                    continue
                cur_stake = Decimal(str(eng.stake_fraction))
                cur_edge = Decimal(str(eng.min_edge))
                target_stake = Decimal(str(p.get("stake_fraction", eng.stake_fraction)))
                target_edge = Decimal(str(p.get("min_edge", eng.min_edge)))

                # bounded step change
                stake_delta = target_stake - cur_stake
                edge_delta = target_edge - cur_edge
                max_s = config.ARCHITECT_MAX_STAKE_STEP
                max_e = config.ARCHITECT_MAX_EDGE_STEP
                max_stake_growth = cur_stake * Decimal("0.5")
                if stake_delta > max_s:
                    target_stake = cur_stake + max_s
                if stake_delta < -max_s:
                    target_stake = cur_stake - max_s
                if target_stake > cur_stake + max_stake_growth:
                    target_stake = cur_stake + max_stake_growth
                if edge_delta > max_e:
                    target_edge = cur_edge + max_e
                if edge_delta < -max_e:
                    target_edge = cur_edge - max_e
                if Decimal(str(states.get(model_id, {}).get("roi_pct", 0))) < Decimal("-5"):
                    target_stake = config.ARCHITECT_MIN_STAKE_FRACTION

                target_stake = self._clip(target_stake, config.ARCHITECT_MIN_STAKE_FRACTION, config.ARCHITECT_MAX_STAKE_FRACTION)
                hard_edge_floor = max(config.ARCHITECT_MIN_EDGE, Decimal("0.02"))
                target_edge = self._clip(target_edge, hard_edge_floor, config.ARCHITECT_MAX_EDGE)
                if target_stake != cur_stake:
                    eng.stake_fraction = float(target_stake)
                    applied = True
                if target_edge != cur_edge:
                    eng.min_edge = float(target_edge)
                    applied = True

        decision = ArchitectDecision(
            ts=now.isoformat(),
            mode=mode,
            applied=applied,
            reason="ok" if proposals else "no_change",
            proposals=proposals,
        )
        self.last_run_ts = now
        self.last_decision = decision
        self._log(decision)
        self._cooldown_multiplier = 2 if applied else 1
        return decision
