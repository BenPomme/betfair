from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, Optional

from funding.portfolios.cascade_alpha.engine import CascadeAlphaEngine
from portfolio.accounting import build_strategy_account
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class CascadeAlphaPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._engine = CascadeAlphaEngine()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _start_engine(self) -> None:
        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._engine.start())
            except Exception:
                logger.exception("Cascade alpha engine crashed")
            finally:
                self._loop = None
                loop.close()
        self._thread = threading.Thread(target=_run, daemon=True, name="cascade-alpha-runner")
        self._thread.start()

    def _stop_engine(self) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._engine.stop(), self._loop)
                future.result(timeout=15.0)
            except Exception:
                logger.exception("Failed stopping cascade alpha engine")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=15.0)
        self._thread = None

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        cascade_contexts = [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") == "binance_cascade_regime"
        ]
        snapshot.update(
            {
                "max_open_positions": 3,
                "max_notional_per_trade_usd": 5000,
                "max_gross_exposure_usd": 30000,
                "max_hold_seconds": 900,
                "daily_loss_limit_pct": 0.03,
                "factory_cascade_strategy_contexts": cascade_contexts,
            }
        )
        return snapshot

    def run(self) -> None:
        self.install_signal_handlers()
        self.initialize_runtime()
        self._start_engine()
        try:
            while not self.should_stop():
                state = self._engine.get_state()
                account = build_strategy_account(
                    portfolio_id=self.spec.portfolio_id,
                    currency=self.spec.currency,
                    starting_balance=self.spec.initial_balance,
                    current_balance=float(state.get("current_balance_usd", self.spec.initial_balance) or self.spec.initial_balance),
                    realized_pnl=float(state.get("realized_pnl_usd", 0.0) or 0.0),
                    unrealized_pnl=float(state.get("unrealized_pnl_usd", 0.0) or 0.0),
                    fees_paid=float(state.get("fees_paid_usd", 0.0) or 0.0),
                    slippage_cost=float(((state.get("execution_quality") or {}).get("avg_modeled_slippage_bps", 0.0)) or 0.0),
                    gross_exposure=float(state.get("gross_exposure_usd", 0.0) or 0.0),
                    wins=sum(1 for t in (state.get("closed_trades") or []) if float(t.get("net_pnl_usd", 0.0) or 0.0) > 0.0),
                    losses=sum(1 for t in (state.get("closed_trades") or []) if float(t.get("net_pnl_usd", 0.0) or 0.0) < 0.0),
                    trade_count=len(state.get("closed_trades") or []),
                    balance_history=state.get("balance_history") or [],
                )
                learner = state.get("learner") or {}
                models = [
                    ModelShadowAccount(
                        portfolio_id=self.spec.portfolio_id,
                        model_id="cascade_online_learner",
                        shadow_starting_balance=self.spec.initial_balance,
                        shadow_current_balance=float(state.get("current_balance_usd", self.spec.initial_balance) or self.spec.initial_balance),
                        shadow_realized_pnl=float(state.get("realized_pnl_usd", 0.0) or 0.0),
                        shadow_roi_pct=((float(state.get("realized_pnl_usd", 0.0) or 0.0) / self.spec.initial_balance) * 100.0) if self.spec.initial_balance else 0.0,
                        settled_count=int(learner.get("settled_count", 0) or 0),
                        metrics=learner,
                        selected_for_execution=bool(learner.get("strict_gate_pass", False)),
                    )
                ]
                self.publish_snapshot(
                    account=account,
                    raw_state=state,
                    readiness=state.get("readiness") or {},
                    models=models,
                    trades=list(state.get("closed_trades") or []) + list(state.get("open_positions") or []),
                    events=list(state.get("events") or []),
                    balance_history=list(state.get("balance_history") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._stop_engine()
            self.finalize_runtime()
