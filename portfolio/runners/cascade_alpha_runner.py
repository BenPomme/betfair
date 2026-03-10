from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, Optional

import config
from funding.portfolios.cascade_alpha.engine import CascadeAlphaEngine
from portfolio.accounting import build_strategy_account
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class CascadeAlphaPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._engine: Optional[CascadeAlphaEngine] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _candidate_contexts(self, snapshot: Dict[str, object]) -> list[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_candidate_contexts", [])
            if item.get("family_id") == "binance_cascade_regime"
        ]

    def _live_contexts(self, snapshot: Dict[str, object]) -> list[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") == "binance_cascade_regime"
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
        max_notional = max(100.0, round(self.spec.initial_balance * max(0.01, min(0.08, selected_stake or 0.03)), 2))
        reduced_multiplier = max(0.25, min(1.0, max_notional / max(1.0, float(getattr(config, "CASCADE_ALPHA_MAX_NOTIONAL_PER_TRADE_USD", 1.0)))))
        overrides: Dict[str, object] = {
            "CASCADE_ALPHA_MAX_NOTIONAL_PER_TRADE_USD": max_notional,
            "CASCADE_ALPHA_POLICY_REDUCED_NOTIONAL_MULTIPLIER": round(reduced_multiplier, 4),
        }
        if selected_horizon > 0:
            overrides["CASCADE_ALPHA_MAX_HOLD_SECONDS"] = max(60, min(3600, selected_horizon))
        if selected_edge > 0.0:
            overrides["CASCADE_ALPHA_POLICY_STRONG_MIN_SIGNAL_SCORE"] = round(max(6.0, min(10.0, 6.0 + (selected_edge * 60.0))), 4)
        return overrides

    def _start_engine(self) -> None:
        self._engine = CascadeAlphaEngine()

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
        cascade_candidate_contexts = self._candidate_contexts(snapshot)
        cascade_live_contexts = self._live_contexts(snapshot)
        cascade_contexts = cascade_live_contexts + cascade_candidate_contexts
        snapshot.update(
            {
                "max_open_positions": int(getattr(config, "CASCADE_ALPHA_MAX_OPEN_POSITIONS", 3)),
                "max_notional_per_trade_usd": float(getattr(config, "CASCADE_ALPHA_MAX_NOTIONAL_PER_TRADE_USD", 5000)),
                "max_gross_exposure_usd": float(getattr(config, "CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD", 30000)),
                "max_hold_seconds": int(getattr(config, "CASCADE_ALPHA_MAX_HOLD_SECONDS", 900)),
                "daily_loss_limit_pct": float(getattr(config, "CASCADE_ALPHA_DAILY_LOSS_LIMIT_PCT", 0.03)),
                "factory_cascade_candidate_contexts": cascade_candidate_contexts,
                "factory_cascade_live_contexts": cascade_live_contexts,
                "factory_cascade_strategy_contexts": cascade_contexts,
            }
        )
        return snapshot

    def run(self) -> None:
        self.install_signal_handlers()
        candidate_context = self.preferred_factory_context(
            family_ids={"binance_cascade_regime"},
            include_live=False,
            include_candidates=True,
        )
        self.apply_factory_config_overrides(
            self._candidate_config_overrides(candidate_context),
            source_context=candidate_context,
        )
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
                    raw_state={**state, **self.factory_applied_runtime()},
                    readiness=state.get("readiness") or {},
                    models=models,
                    trades=list(state.get("closed_trades") or []) + list(state.get("open_positions") or []),
                    events=list(state.get("events") or []),
                    balance_history=list(state.get("balance_history") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._stop_engine()
            self.restore_factory_config_overrides()
            self.finalize_runtime()
