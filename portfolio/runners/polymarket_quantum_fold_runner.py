from __future__ import annotations

import logging
import time
from typing import Dict

import config
from polymarket.engine import PolymarketQuantumFoldEngine
from portfolio.accounting import build_strategy_account
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class PolymarketQuantumFoldPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._engine: PolymarketQuantumFoldEngine | None = None

    def _candidate_contexts(self, snapshot: Dict[str, object]) -> list[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_candidate_contexts", [])
            if item.get("family_id") == "polymarket_cross_venue"
        ]

    def _live_contexts(self, snapshot: Dict[str, object]) -> list[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") == "polymarket_cross_venue"
        ]

    def _candidate_config_overrides(self, context: Dict[str, object] | None) -> Dict[str, object]:
        if not context:
            return {}
        strategy_profile = dict(context.get("strategy_profile") or {})
        selected_stake = float(strategy_profile.get("selected_stake_fraction") or 0.0)
        selected_edge = float(
            strategy_profile.get("artifact_min_edge")
            or strategy_profile.get("selected_min_edge")
            or 0.0
        )
        selected_horizon = int(strategy_profile.get("selected_horizon_seconds") or 0)
        max_notional = max(25.0, round(self.spec.initial_balance * max(0.005, min(0.08, selected_stake or 0.02)), 2))
        max_positions = max(1, min(5, int(round(1 + (selected_stake * 20.0))) if selected_stake > 0 else 2))
        horizon = max(120, min(7200, selected_horizon or int(getattr(config, "POLYMARKET_QF_PRIMARY_HORIZON_SECONDS", 600))))
        label_horizons = ",".join(
            str(value)
            for value in sorted({max(60, horizon // 2), horizon, min(7200, horizon * 3)})
        )
        overrides: Dict[str, object] = {
            "POLYMARKET_QF_MAX_NOTIONAL_PER_TRADE_USD": max_notional,
            "POLYMARKET_QF_MAX_OPEN_POSITIONS": max_positions,
            "POLYMARKET_QF_PRIMARY_HORIZON_SECONDS": horizon,
            "POLYMARKET_QF_LABEL_HORIZONS_SECONDS": label_horizons,
            "POLYMARKET_QF_MAX_HOLD_SECONDS": max(300, horizon),
        }
        if selected_edge > 0.0:
            overrides["POLYMARKET_QF_MIN_EDGE_AFTER_COSTS"] = round(max(0.005, min(0.08, selected_edge)), 6)
        return overrides

    def _build_engine(self) -> PolymarketQuantumFoldEngine:
        return PolymarketQuantumFoldEngine(
            self.store.runtime_dir,
            initial_balance=self.spec.initial_balance,
        )

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        polymarket_candidate_contexts = self._candidate_contexts(snapshot)
        polymarket_live_contexts = self._live_contexts(snapshot)
        polymarket_contexts = polymarket_live_contexts + polymarket_candidate_contexts
        snapshot.update(
            {
                "sports_filter": str(config.POLYMARKET_QF_SPORTS_FILTER),
                "label_horizons_seconds": str(config.POLYMARKET_QF_LABEL_HORIZONS_SECONDS),
                "primary_horizon_seconds": int(config.POLYMARKET_QF_PRIMARY_HORIZON_SECONDS),
                "min_edge_after_costs": float(config.POLYMARKET_QF_MIN_EDGE_AFTER_COSTS),
                "max_open_positions": int(config.POLYMARKET_QF_MAX_OPEN_POSITIONS),
                "max_notional_per_trade_usd": float(config.POLYMARKET_QF_MAX_NOTIONAL_PER_TRADE_USD),
                "stale_quote_seconds": int(config.POLYMARKET_QF_STALE_QUOTE_SECONDS),
                "clob_ws_enabled": bool(config.POLYMARKET_QF_CLOB_WS_ENABLED),
                "factory_polymarket_candidate_contexts": polymarket_candidate_contexts,
                "factory_polymarket_live_contexts": polymarket_live_contexts,
                "factory_polymarket_strategy_contexts": polymarket_contexts,
            }
        )
        return snapshot

    def run(self) -> None:
        self.install_signal_handlers()
        candidate_context = self.preferred_factory_context(
            family_ids={"polymarket_cross_venue"},
            include_live=False,
            include_candidates=True,
        )
        self.apply_factory_config_overrides(
            self._candidate_config_overrides(candidate_context),
            source_context=candidate_context,
        )
        self.initialize_runtime()
        self._engine = self._build_engine()
        self._engine.start()
        try:
            while not self.should_stop():
                try:
                    state = self._engine.step()
                except Exception as exc:
                    logger.exception("Polymarket quantum-fold cycle failed")
                    state = {
                        "portfolio_id": self.spec.portfolio_id,
                        "running": True,
                        "mode": "paper",
                        "status": "error",
                        "error": str(exc),
                        "events": [{"kind": "runner_error", "data": {"error": str(exc), "ts": time.time()}}],
                        "readiness": {
                            "status": "blocked",
                            "blockers": ["runner_cycle_error"],
                            "checks": [
                                {
                                    "name": "runner_cycle_error",
                                    "ok": False,
                                    "reason": str(exc),
                                }
                            ],
                        },
                    }
                account = build_strategy_account(
                    portfolio_id=self.spec.portfolio_id,
                    currency=self.spec.currency,
                    starting_balance=self.spec.initial_balance,
                    current_balance=float(state.get("current_balance_usd", self.spec.initial_balance) or self.spec.initial_balance),
                    realized_pnl=float(state.get("realized_pnl_usd", 0.0) or 0.0),
                    unrealized_pnl=float(self._engine.executor.unrealized_pnl(self._engine.quote_map) if hasattr(self._engine, "quote_map") else 0.0),
                    fees_paid=float(self._engine.executor.fees_paid()),
                    slippage_cost=float(((state.get("execution_quality") or {}).get("avg_modeled_slippage_bps", 0.0)) or 0.0),
                    gross_exposure=float(state.get("gross_exposure_usd", 0.0) or 0.0),
                    wins=sum(1 for trade in (state.get("closed_trades") or []) if float(trade.get("net_pnl_usd", 0.0) or 0.0) > 0.0),
                    losses=sum(1 for trade in (state.get("closed_trades") or []) if float(trade.get("net_pnl_usd", 0.0) or 0.0) < 0.0),
                    trade_count=int(state.get("trade_count", 0) or 0),
                    balance_history=list(state.get("balance_history") or []),
                )
                models = self._engine.model_league.build_accounts()
                trades = list(state.get("closed_trades") or []) + list(state.get("open_positions") or [])
                self.publish_snapshot(
                    account=account,
                    raw_state={**state, **self.factory_applied_runtime()},
                    readiness=state.get("readiness") or {},
                    models=models,
                    trades=trades,
                    events=list(state.get("events") or []),
                    balance_history=list(state.get("balance_history") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            if self._engine is not None:
                self._engine.stop()
            self.restore_factory_config_overrides()
            self.finalize_runtime()
