from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

import config
from funding.portfolios.contrarian_legacy.engine import ContrarianLegacyEngine
from portfolio.accounting import build_strategy_account
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class ContrarianLegacyPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        runtime_dir = self.store.runtime_dir
        self._engine = ContrarianLegacyEngine(
            state_path=str(runtime_dir / "directional_positions.json"),
            trade_log_path=str(runtime_dir / "contrarian_trade_log.jsonl"),
            quality_state_path=str(runtime_dir / "contrarian_online_learner_quality.json"),
        )
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
                logger.exception("Contrarian legacy engine crashed")
            finally:
                self._loop = None
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="contrarian-legacy-runner")
        self._thread.start()

    def _stop_engine(self) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._engine.stop(), self._loop)
                future.result(timeout=15.0)
            except Exception:
                logger.exception("Failed stopping contrarian legacy engine")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=15.0)
        self._thread = None

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        funding_contexts = [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") == "binance_funding_contrarian"
        ]
        snapshot.update(
            {
                "max_positions": int(config.CONTRARIAN_MAX_POSITIONS),
                "min_confidence": float(config.CONTRARIAN_MIN_CONFIDENCE),
                "max_hold_hours": int(config.CONTRARIAN_MAX_HOLD_HOURS),
                "capital_per_trade_pct": float(config.CONTRARIAN_CAPITAL_PER_TRADE_PCT),
                "daily_loss_limit_pct": float(config.CONTRARIAN_DAILY_LOSS_LIMIT_PCT),
                "factory_funding_strategy_contexts": funding_contexts,
                "factory_funding_model_meta_path": next(
                    (
                        str((item.get("artifact_refs") or {}).get("model_meta"))
                        for item in funding_contexts
                        if (item.get("artifact_refs") or {}).get("model_meta")
                    ),
                    None,
                ),
            }
        )
        return snapshot

    def _build_model_accounts(self, state: Dict[str, object]) -> List[ModelShadowAccount]:
        learner = state.get("online_learner") or {}
        rolling = learner.get("rolling_200") or {}
        current_balance = float(state.get("current_balance_usd", self.spec.initial_balance) or self.spec.initial_balance)
        realized_pnl = float(state.get("total_realized_pnl_usd", 0.0) or 0.0)
        return [
            ModelShadowAccount(
                portfolio_id=self.spec.portfolio_id,
                model_id="contrarian_online_learner",
                shadow_starting_balance=self.spec.initial_balance,
                shadow_current_balance=current_balance,
                shadow_realized_pnl=realized_pnl,
                shadow_roi_pct=float(rolling.get("roi_pct", 0.0) or 0.0),
                settled_count=int(rolling.get("settled", 0) or 0),
                metrics=dict(learner),
                selected_for_execution=bool(learner.get("strict_gate_pass", False)),
            )
        ]

    def run(self) -> None:
        self.install_signal_handlers()
        self.initialize_runtime()
        self._start_engine()
        try:
            while not self.should_stop():
                state = self._engine.get_state()
                current_balance = float(state.get("current_balance_usd", self.spec.initial_balance) or self.spec.initial_balance)
                account = build_strategy_account(
                    portfolio_id=self.spec.portfolio_id,
                    currency=self.spec.currency,
                    starting_balance=self.spec.initial_balance,
                    current_balance=current_balance,
                    realized_pnl=float(state.get("total_realized_pnl_usd", 0.0) or 0.0),
                    unrealized_pnl=float(state.get("unrealized_pnl_usd", 0.0) or 0.0),
                    fees_paid=float(state.get("total_fees_paid_usd", 0.0) or 0.0),
                    gross_exposure=float(((state.get("risk") or {}).get("gross_exposure_usd", 0.0)) or 0.0),
                    wins=int(state.get("wins", 0) or 0),
                    losses=int(state.get("losses", 0) or 0),
                    trade_count=int(state.get("trade_count", 0) or 0),
                )
                self.publish_snapshot(
                    account=account,
                    raw_state=state,
                    readiness=state.get("readiness") or {},
                    models=self._build_model_accounts(state),
                    trades=list(state.get("all_positions") or []),
                    events=list(state.get("events") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._stop_engine()
            self.finalize_runtime()
