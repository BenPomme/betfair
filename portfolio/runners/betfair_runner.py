from __future__ import annotations

import logging
import time
from typing import Dict, List

import config
from monitoring.engine import TradingEngine
from monitoring.live_readiness import evaluate_betfair_live_readiness
from portfolio.accounting import build_strategy_account, normalize_balance_history
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec

logger = logging.getLogger(__name__)


class BetfairPortfolioRunner(PortfolioRunnerBase):
    def __init__(self, spec: PortfolioRunnerSpec):
        super().__init__(spec)
        self._engine = TradingEngine()

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        snapshot.update(
            {
                "paper_trading": bool(config.PAPER_TRADING),
                "initial_balance_eur": float(config.INITIAL_BALANCE_EUR),
                "stake_fraction": float(config.STAKE_FRACTION),
                "prediction_enabled": bool(config.PREDICTION_ENABLED),
                "paper_state_path": str(self.store.runtime_dir / "paper_executor_state.json"),
                "paper_trades_log_path": str(self.store.runtime_dir / "paper_trades.jsonl"),
            }
        )
        return snapshot

    def _build_model_accounts(self, state: Dict[str, object]) -> List[ModelShadowAccount]:
        models = []
        prediction_models = state.get("prediction_models") or {}
        for model_id, payload in prediction_models.items():
            if not isinstance(payload, dict):
                continue
            balance = float(payload.get("balance_eur", config.PREDICTION_INITIAL_BALANCE_EUR) or config.PREDICTION_INITIAL_BALANCE_EUR)
            starting = float(config.PREDICTION_INITIAL_BALANCE_EUR)
            models.append(
                ModelShadowAccount(
                    portfolio_id=self.spec.portfolio_id,
                    model_id=str(model_id),
                    shadow_starting_balance=starting,
                    shadow_current_balance=balance,
                    shadow_realized_pnl=balance - starting,
                    shadow_roi_pct=float(payload.get("roi_pct", 0.0) or 0.0),
                    settled_count=int(payload.get("settled_bets", 0) or 0),
                    metrics=dict(payload),
                    selected_for_execution=bool(payload.get("strict_gate_pass", False)),
                )
            )
        return models

    def run(self) -> None:
        self.install_signal_handlers()
        self.initialize_runtime()

        config.PAPER_STATE_PATH = str(self.store.runtime_dir / "paper_executor_state.json")
        config.PAPER_TRADES_LOG_PATH = str(self.store.runtime_dir / "paper_trades.jsonl")

        try:
            started = self._engine.start()
            if not started.get("ok"):
                state = {
                    "portfolio_id": self.spec.portfolio_id,
                    "running": False,
                    "status": "error",
                    "error": started.get("error", "betfair_start_failed"),
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

            while not self.should_stop():
                state = self._engine.get_state()
                balance_history = normalize_balance_history(state.get("balance_history") or [])
                current_balance = float(state.get("balance_eur", self.spec.initial_balance) or self.spec.initial_balance)
                trades = list(state.get("trades") or [])
                wins = sum(1 for trade in trades if float(trade.get("net_profit_eur", 0.0) or 0.0) > 0.0)
                losses = sum(1 for trade in trades if float(trade.get("net_profit_eur", 0.0) or 0.0) < 0.0)
                account = build_strategy_account(
                    portfolio_id=self.spec.portfolio_id,
                    currency=self.spec.currency,
                    starting_balance=self.spec.initial_balance,
                    current_balance=current_balance,
                    realized_pnl=current_balance - self.spec.initial_balance,
                    wins=wins,
                    losses=losses,
                    trade_count=int((state.get("session") or {}).get("trade_count", len(trades)) or len(trades)),
                    balance_history=balance_history,
                )
                readiness = evaluate_betfair_live_readiness(state)
                enriched_state = dict(state)
                enriched_state.update(
                    {
                        "portfolio_id": self.spec.portfolio_id,
                        "status": "running" if state.get("running") else "idle",
                        "control_mode": self.spec.control_mode,
                    }
                )
                self.publish_snapshot(
                    account=account,
                    raw_state=enriched_state,
                    readiness=readiness,
                    models=self._build_model_accounts(state),
                    trades=trades,
                    events=list(state.get("events") or []),
                    balance_history=balance_history,
                )
                time.sleep(self._heartbeat_seconds)
        finally:
            try:
                self._engine.stop()
            except Exception:
                logger.exception("Failed to stop Betfair engine cleanly")
            self.finalize_runtime()
