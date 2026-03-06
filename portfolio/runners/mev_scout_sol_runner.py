from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from onchain.solana.mev_scout.engine import MevScoutEngine
from portfolio.accounting import build_strategy_account
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class MevScoutSolPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._engine = MevScoutEngine()
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
                logger.exception("MEV scout engine crashed")
            finally:
                self._loop = None
                loop.close()
        self._thread = threading.Thread(target=_run, daemon=True, name="mev-scout-sol-runner")
        self._thread.start()

    def _stop_engine(self) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._engine.stop(), self._loop)
                future.result(timeout=10.0)
            except Exception:
                logger.exception("Failed stopping MEV scout engine")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._thread = None

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
                    current_balance=float(state.get("shadow_balance_usd", self.spec.initial_balance) or self.spec.initial_balance),
                    realized_pnl=float(state.get("shadow_realized_pnl_usd", 0.0) or 0.0),
                    trade_count=int(state.get("opportunity_count", 0) or 0),
                    balance_history=state.get("balance_history") or [],
                )
                model = ModelShadowAccount(
                    portfolio_id=self.spec.portfolio_id,
                    model_id="whale_follow_shadow_v1",
                    shadow_starting_balance=self.spec.initial_balance,
                    shadow_current_balance=float(state.get("shadow_balance_usd", self.spec.initial_balance) or self.spec.initial_balance),
                    shadow_realized_pnl=float(state.get("shadow_realized_pnl_usd", 0.0) or 0.0),
                    shadow_roi_pct=((float(state.get("shadow_realized_pnl_usd", 0.0) or 0.0) / self.spec.initial_balance) * 100.0) if self.spec.initial_balance else 0.0,
                    settled_count=int(state.get("opportunity_count", 0) or 0),
                    metrics={"latency_ms": state.get("latency_ms"), "provider_configured": state.get("provider_configured")},
                    selected_for_execution=False,
                )
                self.publish_snapshot(
                    account=account,
                    raw_state=state,
                    readiness=state.get("readiness") or {},
                    models=[model],
                    trades=list(state.get("opportunities") or []),
                    events=list(state.get("events") or []),
                    balance_history=list(state.get("balance_history") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._stop_engine()
            self.finalize_runtime()
