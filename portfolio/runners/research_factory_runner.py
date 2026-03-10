from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import config
from factory.execution_bridge import FactoryExecutionBridge
from factory.orchestrator import FactoryOrchestrator
from portfolio.accounting import build_strategy_account, utc_now_iso
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class ResearchFactoryPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._orchestrator = FactoryOrchestrator(Path(__file__).resolve().parents[2])
        self._execution_bridge = FactoryExecutionBridge()
        self._balance_history: List[Dict[str, object]] = self.store.read_balance_history()
        self._last_cycle_started_at: float = 0.0
        self._cycle_interval = max(15, int(getattr(config, "FACTORY_CYCLE_INTERVAL_SECONDS", 60)))

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        snapshot.update(
            {
                "factory_root": str(self._orchestrator.registry.root),
                "goldfish_root": str(self._orchestrator.bridge.root),
                "goldfish_mode": "sidecar",
                "compute_posture": "local_first_hybrid",
                "promotion_authority": "human_signoff",
                "paper_gate_monthly_roi_pct": float(getattr(config, "FACTORY_PAPER_GATE_MONTHLY_ROI_PCT", 5.0)),
                "paper_gate_max_drawdown_pct": float(getattr(config, "FACTORY_PAPER_GATE_MAX_DRAWDOWN_PCT", 8.0)),
                "execution_bridge_autostart_enabled": bool(getattr(config, "FACTORY_EXECUTION_AUTOSTART_ENABLED", True)),
            }
        )
        return snapshot

    def _append_balance_point(self, balance: float) -> None:
        point = {"ts": utc_now_iso(), "balance": round(float(balance), 6)}
        if self._balance_history:
            last = self._balance_history[-1]
            if abs(float(last.get("balance", 0.0)) - point["balance"]) < 1e-9:
                return
        self._balance_history.append(point)
        self._balance_history = self._balance_history[-1000:]

    def _build_model_accounts(self, state: Dict[str, object]) -> List[ModelShadowAccount]:
        models: List[ModelShadowAccount] = []
        for row in sorted(
            list(state.get("lineages") or []),
            key=lambda item: (
                item.get("pareto_rank") if item.get("pareto_rank") is not None else 999,
                -float(item.get("fitness_score", 0.0) or 0.0),
            ),
        )[:12]:
            budget_weight = float(row.get("budget_weight_pct", 0.0) or 0.0)
            shadow_starting_balance = round((self.spec.initial_balance * budget_weight) / 100.0, 6)
            shadow_realized_pnl = round(float(row.get("net_pnl", 0.0) or 0.0), 6)
            metrics = {
                "model_kind": row.get("role"),
                "current_stage": row.get("current_stage"),
                "pareto_rank": row.get("pareto_rank"),
                "fitness_score": row.get("fitness_score"),
                "learning_settled": int(row.get("settled_count", 0) or 0),
                "settled_count": int(row.get("settled_count", 0) or 0),
                "learning_updates": int(state.get("cycle_count", 0) or 0),
                "model_updates": int(state.get("cycle_count", 0) or 0),
                "recent_learning_brier_lift": float(row.get("calibration_lift_abs", 0.0) or 0.0),
                "strict_gate_pass": bool(row.get("strict_gate_pass", False)),
                "strict_gate_reason": (row.get("blockers") or ["none"])[0],
                "paper_days": int(row.get("paper_days", 0) or 0),
                "tweak_count": int(row.get("tweak_count", 0) or 0),
                "max_tweaks": int(row.get("max_tweaks", 2) or 2),
                "iteration_status": str(row.get("iteration_status") or ""),
                "lead_agent_role": str(row.get("lead_agent_role") or ""),
                "collaborating_agent_roles": list(row.get("collaborating_agent_roles") or []),
            }
            models.append(
                ModelShadowAccount(
                    portfolio_id=self.spec.portfolio_id,
                    model_id=str(row.get("lineage_id") or "unknown"),
                    shadow_starting_balance=shadow_starting_balance,
                    shadow_current_balance=round(shadow_starting_balance + shadow_realized_pnl, 6),
                    shadow_realized_pnl=shadow_realized_pnl,
                    shadow_roi_pct=float(row.get("monthly_roi_pct", 0.0) or 0.0),
                    settled_count=int(row.get("settled_count", 0) or 0),
                    metrics=metrics,
                    selected_for_execution=str(row.get("current_stage")) == "approved_live",
                )
            )
        return models

    def _trades_from_lineages(self, state: Dict[str, object]) -> List[Dict[str, object]]:
        trades: List[Dict[str, object]] = []
        for row in state.get("lineages") or []:
            trades.append(
                {
                    "trade_id": row.get("lineage_id"),
                    "status": row.get("current_stage"),
                    "net_pnl_usd": row.get("net_pnl", 0.0),
                    "selection": row.get("label"),
                    "reason": (row.get("blockers") or ["ready"])[0],
                    "updated_at": state.get("last_cycle_at"),
                }
            )
        return trades[:50]

    def _events(self) -> List[Dict[str, object]]:
        history_path = self._orchestrator.registry.history_dir / "promotions.jsonl"
        if not history_path.exists():
            return []
        rows = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return [
            {
                "kind": "factory_transition",
                "data": row,
            }
            for row in rows[-50:]
        ]

    def run(self) -> None:
        self.install_signal_handlers()
        self.initialize_runtime()
        try:
            while not self.should_stop():
                now = time.time()
                if not self._last_cycle_started_at or (now - self._last_cycle_started_at) >= self._cycle_interval:
                    self._last_cycle_started_at = now
                    state = self._orchestrator.run_cycle()
                else:
                    state = self._orchestrator.get_state()
                execution_bridge = self._execution_bridge.sync(state)
                summary = dict(state.get("research_summary") or {})
                realized = float(summary.get("paper_pnl", 0.0) or 0.0)
                current_balance = self.spec.initial_balance + realized
                self._append_balance_point(current_balance)
                account = build_strategy_account(
                    portfolio_id=self.spec.portfolio_id,
                    currency=self.spec.currency,
                    starting_balance=self.spec.initial_balance,
                    current_balance=current_balance,
                    realized_pnl=realized,
                    trade_count=len(state.get("lineages") or []),
                    balance_history=self._balance_history,
                )
                enriched_state = dict(state)
                enriched_state.update(
                    {
                        "portfolio_id": self.spec.portfolio_id,
                        "mode": "research",
                        "factory_manifest_count": len((state.get("manifests") or {}).get("live_loadable") or []),
                        "execution_bridge": execution_bridge,
                    }
                )
                self.publish_snapshot(
                    account=account,
                    raw_state=enriched_state,
                    readiness=dict(state.get("readiness") or {}),
                    models=self._build_model_accounts(state),
                    trades=self._trades_from_lineages(state),
                    events=self._events(),
                    balance_history=self._balance_history,
                )
                time.sleep(self._heartbeat_seconds)
        except Exception:
            logger.exception("Research factory runner failed")
            raise
        finally:
            self.finalize_runtime()
