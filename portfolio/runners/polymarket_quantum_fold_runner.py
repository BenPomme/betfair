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
        self._engine = PolymarketQuantumFoldEngine(
            self.store.runtime_dir,
            initial_balance=self.spec.initial_balance,
        )

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        snapshot.update(
            {
                "sports_filter": str(config.POLYMARKET_QF_SPORTS_FILTER),
                "label_horizons_seconds": str(config.POLYMARKET_QF_LABEL_HORIZONS_SECONDS),
                "primary_horizon_seconds": int(config.POLYMARKET_QF_PRIMARY_HORIZON_SECONDS),
                "min_edge_after_costs": float(config.POLYMARKET_QF_MIN_EDGE_AFTER_COSTS),
                "max_open_positions": int(config.POLYMARKET_QF_MAX_OPEN_POSITIONS),
                "max_notional_per_trade_usd": float(config.POLYMARKET_QF_MAX_NOTIONAL_PER_TRADE_USD),
                "stale_quote_seconds": int(config.POLYMARKET_QF_STALE_QUOTE_SECONDS),
                "match_markets_only": bool(config.POLYMARKET_QF_MATCH_MARKETS_ONLY),
                "max_settlement_hours": int(config.POLYMARKET_QF_MAX_SETTLEMENT_HOURS),
                "clob_ws_enabled": bool(config.POLYMARKET_QF_CLOB_WS_ENABLED),
            }
        )
        return snapshot

    def run(self) -> None:
        self.install_signal_handlers()
        self.initialize_runtime()
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
                    raw_state=state,
                    readiness=state.get("readiness") or {},
                    models=models,
                    trades=trades,
                    events=list(state.get("events") or []),
                    balance_history=list(state.get("balance_history") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._engine.stop()
            self.finalize_runtime()
