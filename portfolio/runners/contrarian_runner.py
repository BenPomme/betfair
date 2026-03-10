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
        self._state_path = str(runtime_dir / "directional_positions.json")
        self._trade_log_path = str(runtime_dir / "contrarian_trade_log.jsonl")
        self._quality_state_path = str(runtime_dir / "contrarian_online_learner_quality.json")
        self._engine: Optional[ContrarianLegacyEngine] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _candidate_contexts(self, snapshot: Dict[str, object]) -> List[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_candidate_contexts", [])
            if item.get("family_id") == "binance_funding_contrarian"
        ]

    def _live_contexts(self, snapshot: Dict[str, object]) -> List[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") == "binance_funding_contrarian"
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
        overrides: Dict[str, object] = {}
        if selected_stake > 0.0:
            overrides["CONTRARIAN_CAPITAL_PER_TRADE_PCT"] = round(max(0.005, min(0.08, selected_stake)), 6)
        if selected_edge > 0.0:
            overrides["CONTRARIAN_MIN_CONFIDENCE"] = round(max(0.55, min(0.85, 0.55 + (selected_edge * 4.0))), 6)
        if selected_horizon > 0:
            overrides["CONTRARIAN_MAX_HOLD_HOURS"] = max(1, min(48, int(round(selected_horizon / 3600.0))))
        return overrides

    def _start_engine(self) -> None:
        self._engine = ContrarianLegacyEngine(
            state_path=self._state_path,
            trade_log_path=self._trade_log_path,
            quality_state_path=self._quality_state_path,
        )

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
        funding_candidate_contexts = self._candidate_contexts(snapshot)
        funding_live_contexts = self._live_contexts(snapshot)
        funding_contexts = funding_live_contexts + funding_candidate_contexts
        snapshot.update(
            {
                "max_positions": int(config.CONTRARIAN_MAX_POSITIONS),
                "min_confidence": float(config.CONTRARIAN_MIN_CONFIDENCE),
                "max_hold_hours": int(config.CONTRARIAN_MAX_HOLD_HOURS),
                "capital_per_trade_pct": float(config.CONTRARIAN_CAPITAL_PER_TRADE_PCT),
                "daily_loss_limit_pct": float(config.CONTRARIAN_DAILY_LOSS_LIMIT_PCT),
                "factory_funding_candidate_contexts": funding_candidate_contexts,
                "factory_funding_live_contexts": funding_live_contexts,
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
        candidate_context = self.preferred_factory_context(
            family_ids={"binance_funding_contrarian"},
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
                    raw_state={**state, **self.factory_applied_runtime()},
                    readiness=state.get("readiness") or {},
                    models=self._build_model_accounts(state),
                    trades=list(state.get("all_positions") or []),
                    events=list(state.get("events") or []),
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._stop_engine()
            self.restore_factory_config_overrides()
            self.finalize_runtime()
