from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

import config
from funding.main import FundingEngine
from monitoring.live_readiness import evaluate_binance_live_readiness
from portfolio.accounting import build_strategy_account, utc_now_iso
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class HedgeValidationPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._engine: Optional[FundingEngine] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._balance_history: List[Dict[str, object]] = self.store.read_balance_history()

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        funding_candidate_contexts = [
            item for item in snapshot.get("factory_candidate_contexts", [])
            if item.get("family_id") == "binance_funding_contrarian"
        ]
        funding_live_contexts = [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") == "binance_funding_contrarian"
        ]
        snapshot.update(
            {
                "funding_mode": config.FUNDING_MODE,
                "validation_mode": True,
                "validation_scope": "hedge_only",
                "execution_mode": "fail_closed",
                "max_total_exposure_usd": float(config.FUNDING_MAX_TOTAL_EXPOSURE_USD),
                "max_open_hedges": int(config.FUNDING_MAX_OPEN_HEDGES),
                "shared_learner_read_only": True,
                "state_path": str(config.FUNDING_STATE_PATH),
                "factory_funding_candidate_contexts": funding_candidate_contexts,
                "factory_funding_live_contexts": funding_live_contexts,
                "factory_funding_strategy_contexts": funding_live_contexts + funding_candidate_contexts,
            }
        )
        return snapshot

    def _start_engine(self) -> Dict[str, object]:
        config.FUNDING_VALIDATION_MODE = True
        config.FUNDING_VALIDATION_SCOPE = "hedge_only"
        config.FUNDING_PAPER_REQUIRE_TESTNET_FILLS = True
        config.FUNDING_PAPER_ALLOW_SIM_FALLBACK = False
        config.FUNDING_SHARED_LEARNER_READ_ONLY = True
        config.CONTRARIAN_ENABLED = False

        self._engine = FundingEngine()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._engine.start())
            except Exception:
                logger.exception("Hedge funding engine crashed")
            finally:
                self._loop = None
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="hedge-validation-runner")
        self._thread.start()
        return {"ok": True}

    def _stop_engine(self) -> None:
        if self._engine is None:
            return
        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._engine.stop(), self._loop)
                future.result(timeout=15.0)
            except Exception:
                logger.exception("Failed stopping hedge engine gracefully")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=15.0)
        self._thread = None
        self._engine = None

    def _append_balance_point(self, balance: float) -> None:
        point = {"ts": utc_now_iso(), "balance": round(float(balance), 6)}
        if self._balance_history:
            last = self._balance_history[-1]
            if abs(float(last.get("balance", 0.0)) - point["balance"]) < 1e-9:
                return
        self._balance_history.append(point)
        self._balance_history = self._balance_history[-1000:]

    def _build_model_accounts(self, state: Dict[str, object]) -> List[ModelShadowAccount]:
        learner = state.get("online_learner") or {}
        rolling = learner.get("rolling_200") or {}
        return [
            ModelShadowAccount(
                portfolio_id=self.spec.portfolio_id,
                model_id="funding_online_learner",
                shadow_starting_balance=self.spec.initial_balance,
                shadow_current_balance=self.spec.initial_balance + float(state.get("realized_net_pnl_usd", 0.0) or 0.0),
                shadow_realized_pnl=float(state.get("realized_net_pnl_usd", 0.0) or 0.0),
                shadow_roi_pct=float(rolling.get("roi_pct", state.get("realized_roi_pct", 0.0)) or 0.0),
                settled_count=int(rolling.get("settled", 0) or 0),
                metrics=dict(learner),
                selected_for_execution=True,
            )
        ]

    def run(self) -> None:
        self.install_signal_handlers()
        candidate_context = self.preferred_factory_context(
            family_ids={"binance_funding_contrarian"},
            include_live=False,
            include_candidates=True,
        )
        self.apply_factory_config_overrides({}, source_context=candidate_context)
        self.initialize_runtime()

        started = self._start_engine()
        if not started.get("ok"):
            state = {
                "portfolio_id": self.spec.portfolio_id,
                "running": False,
                "status": "error",
                "error": started.get("error", "hedge_start_failed"),
            }
            account = build_strategy_account(
                portfolio_id=self.spec.portfolio_id,
                currency=self.spec.currency,
                starting_balance=self.spec.initial_balance,
                current_balance=self.spec.initial_balance,
                realized_pnl=0.0,
            )
            self.publish_snapshot(account=account, raw_state=state, readiness={"status": "error", "blockers": [state["error"]]})
            return

        try:
            while not self.should_stop():
                if self._engine is None:
                    break
                state = self._engine.get_state()
                realized = float(state.get("realized_net_pnl_usd", 0.0) or 0.0)
                unrealized = float(state.get("estimated_next_funding_total", 0.0) or 0.0)
                current_balance = self.spec.initial_balance + realized + unrealized
                self._append_balance_point(current_balance)
                closed_positions = [p for p in (state.get("all_positions") or []) if str(p.get("status", "")).upper() == "CLOSED"]
                wins = 0
                losses = 0
                for position in closed_positions:
                    try:
                        net = float(position.get("funding_collected", 0.0) or 0.0) + float(position.get("exit_pnl", 0.0) or 0.0) - float(position.get("trading_fees_paid", 0.0) or 0.0)
                    except Exception:
                        net = 0.0
                    if net > 0:
                        wins += 1
                    elif net < 0:
                        losses += 1
                account = build_strategy_account(
                    portfolio_id=self.spec.portfolio_id,
                    currency=self.spec.currency,
                    starting_balance=self.spec.initial_balance,
                    current_balance=current_balance,
                    realized_pnl=realized,
                    unrealized_pnl=unrealized,
                    fees_paid=float(state.get("total_fees_paid", 0.0) or 0.0),
                    slippage_cost=float(((state.get("cost_breakdown") or {}).get("estimated_slippage_cost_usd", 0.0)) or 0.0),
                    gross_exposure=float(state.get("total_exposure", 0.0) or 0.0),
                    wins=wins,
                    losses=losses,
                    trade_count=int(state.get("closed_hedges", 0) or 0),
                    balance_history=self._balance_history,
                )
                readiness = state.get("readiness_v2") or evaluate_binance_live_readiness(state)
                enriched_state = dict(state)
                enriched_state.update(
                    {
                        "portfolio_id": self.spec.portfolio_id,
                        "status": "running" if state.get("running") else "idle",
                        "control_mode": self.spec.control_mode,
                        **self.factory_applied_runtime(),
                    }
                )
                events = list((state.get("settlement_audit") or {}).get("recent", [])) + list(state.get("learning_events") or [])
                self.publish_snapshot(
                    account=account,
                    raw_state=enriched_state,
                    readiness=readiness,
                    models=self._build_model_accounts(state),
                    trades=list(state.get("all_positions") or []),
                    events=events[-200:],
                    balance_history=self._balance_history,
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            self._stop_engine()
            self.restore_factory_config_overrides()
            self.finalize_runtime()
