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

    def _candidate_contexts(self, snapshot: Dict[str, object]) -> List[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_candidate_contexts", [])
            if item.get("family_id") in {"betfair_prediction_value_league", "betfair_information_lag"}
        ]

    def _live_contexts(self, snapshot: Dict[str, object]) -> List[Dict[str, object]]:
        return [
            item for item in snapshot.get("factory_live_contexts", [])
            if item.get("family_id") in {"betfair_prediction_value_league", "betfair_information_lag"}
        ]

    def _candidate_config_overrides(self, context: Dict[str, object] | None) -> Dict[str, object]:
        if not context:
            return {}
        strategy_profile = dict(context.get("strategy_profile") or {})
        artifact_payloads = dict(context.get("artifact_payloads") or {})
        train_payload = dict(artifact_payloads.get("train") or {})
        requested_model_class = str(
            train_payload.get("requested_model_class")
            or strategy_profile.get("requested_model_class")
            or strategy_profile.get("selected_model_class")
            or ""
        ).strip().lower()
        model_kind = None
        if requested_model_class in {"hybrid_logit", "logit", "probit", "transformer", "lstm", "gru", "sequence"}:
            model_kind = "hybrid_logit"
        elif requested_model_class in {"market_calibrated", "calibrated", "xgboost", "gbdt", "tree", "forest"}:
            model_kind = "market_calibrated"
        min_edge = float(
            train_payload.get("min_edge")
            or strategy_profile.get("artifact_min_edge")
            or strategy_profile.get("selected_min_edge")
            or 0.0
        )
        stake_fraction = float(strategy_profile.get("selected_stake_fraction") or 0.0)
        overrides: Dict[str, object] = {}
        if min_edge > 0.0:
            overrides["PREDICTION_MIN_EDGE"] = round(max(0.01, min(0.15, min_edge)), 6)
        if stake_fraction > 0.0:
            clipped_stake = round(max(0.01, min(0.25, stake_fraction)), 6)
            overrides["PREDICTION_STAKE_FRACTION"] = clipped_stake
            overrides["STAKE_FRACTION"] = round(max(0.01, min(0.20, clipped_stake)), 6)
        if model_kind:
            overrides["PREDICTION_MODEL_KINDS"] = model_kind
        return overrides

    def build_config_snapshot(self) -> Dict[str, object]:
        snapshot = super().build_config_snapshot()
        betfair_candidate_contexts = self._candidate_contexts(snapshot)
        betfair_live_contexts = self._live_contexts(snapshot)
        betfair_contexts = betfair_live_contexts + betfair_candidate_contexts
        snapshot.update(
            {
                "paper_trading": bool(config.PAPER_TRADING),
                "initial_balance_eur": float(config.INITIAL_BALANCE_EUR),
                "stake_fraction": float(config.STAKE_FRACTION),
                "prediction_enabled": bool(config.PREDICTION_ENABLED),
                "paper_state_path": str(self.store.runtime_dir / "paper_executor_state.json"),
                "paper_trades_log_path": str(self.store.runtime_dir / "paper_trades.jsonl"),
                "factory_betfair_candidate_contexts": betfair_candidate_contexts,
                "factory_betfair_live_contexts": betfair_live_contexts,
                "factory_betfair_strategy_contexts": betfair_contexts,
                "factory_prediction_policy_gate_path": next(
                    (
                        str((item.get("artifact_refs") or {}).get("policy_gate"))
                        for item in betfair_contexts
                        if (item.get("artifact_refs") or {}).get("policy_gate")
                    ),
                    None,
                ),
            }
        )
        return snapshot

    def _build_model_accounts(self, state: Dict[str, object]) -> List[ModelShadowAccount]:
        models = []
        prediction_models = state.get("prediction_models") or {}
        for model_id, payload in prediction_models.items():
            if not isinstance(payload, dict):
                continue
            starting = float(payload.get("initial_balance_eur", config.PREDICTION_INITIAL_BALANCE_EUR) or config.PREDICTION_INITIAL_BALANCE_EUR)
            realized_pnl = float(payload.get("total_pnl_eur", 0.0) or 0.0)
            equity = starting + realized_pnl
            metrics = dict(payload)
            metrics.setdefault("cash_balance_eur", float(payload.get("balance_eur", starting) or starting))
            models.append(
                ModelShadowAccount(
                    portfolio_id=self.spec.portfolio_id,
                    model_id=str(model_id),
                    shadow_starting_balance=starting,
                    shadow_current_balance=equity,
                    shadow_realized_pnl=realized_pnl,
                    shadow_roi_pct=float(payload.get("roi_pct", 0.0) or 0.0),
                    settled_count=int(payload.get("settled_bets", 0) or 0),
                    metrics=metrics,
                    selected_for_execution=bool(payload.get("strict_gate_pass", False)),
                )
            )
        return models

    def _prediction_league_summary(self, state: Dict[str, object]) -> Dict[str, object]:
        prediction_models = state.get("prediction_models") or {}
        models = [payload for payload in prediction_models.values() if isinstance(payload, dict)]
        if not models:
            return {
                "model_count": 0,
                "active_models": 0,
                "realized_pnl_eur": 0.0,
                "avg_roi_pct": 0.0,
                "open_positions": 0,
                "settled_bets": 0,
                "learning_settled": 0,
                "best_model_id": None,
                "best_model_pnl_eur": 0.0,
            }
        best_model = max(models, key=lambda item: float(item.get("total_pnl_eur", 0.0) or 0.0))
        active_models = sum(1 for item in models if int(item.get("open_positions", 0) or 0) > 0 or int(item.get("settled_bets", 0) or 0) > 0)
        return {
            "model_count": len(models),
            "active_models": active_models,
            "realized_pnl_eur": round(sum(float(item.get("total_pnl_eur", 0.0) or 0.0) for item in models), 4),
            "avg_roi_pct": round(sum(float(item.get("roi_pct", 0.0) or 0.0) for item in models) / len(models), 4),
            "open_positions": int(sum(int(item.get("open_positions", 0) or 0) for item in models)),
            "settled_bets": int(sum(int(item.get("settled_bets", 0) or 0) for item in models)),
            "learning_settled": int(sum(int(item.get("learning_settled", 0) or 0) for item in models)),
            "best_model_id": best_model.get("model_id"),
            "best_model_pnl_eur": float(best_model.get("total_pnl_eur", 0.0) or 0.0),
        }

    def run(self) -> None:
        self.install_signal_handlers()
        candidate_context = self.preferred_factory_context(
            family_ids={"betfair_prediction_value_league", "betfair_information_lag"},
            include_live=False,
            include_candidates=True,
        )
        self.apply_factory_config_overrides(
            self._candidate_config_overrides(candidate_context),
            source_context=candidate_context,
        )
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
                        "prediction_league_summary": self._prediction_league_summary(state),
                        **self.factory_applied_runtime(),
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
            self.restore_factory_config_overrides()
            self.finalize_runtime()
