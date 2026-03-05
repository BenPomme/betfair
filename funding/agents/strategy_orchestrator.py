"""
Strategy Orchestrator — Zero-Touch ML Lifecycle Manager.

Autonomous agent that manages the full contrarian/regime model lifecycle:
COLLECTING → TRAINING → ACTIVATING → MONITORING → (retrain/disable)

Deterministic (rules-only) monitoring decisions (no local LLM/Ollama dependency).
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# States
COLLECTING = "COLLECTING"
TRAINING = "TRAINING"
ACTIVATING = "ACTIVATING"
MONITORING = "MONITORING"

# Minimum symbols with sufficient data before training
_MIN_SYMBOLS_WITH_DATA = 2


@dataclass
class OrchestratorDecision:
    ts: str
    state: str
    action: str
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class StrategyOrchestrator:
    """Autonomous ML lifecycle manager for funding-rate strategies."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._state: str = COLLECTING
        self._interval = max(60, config.ORCHESTRATOR_INTERVAL_SECONDS)

        # Training metrics
        self._last_train_ts: Optional[datetime] = None
        self._last_train_auc: float = 0.0
        self._train_count: int = 0
        self._consecutive_train_failures: int = 0
        self._active_contrarian_model: Optional[str] = None
        self._active_regime_model: Optional[str] = None
        self._last_training_error: Optional[str] = None  # reason when training failed or AUC too low

        # JSONL log
        self._log_dir = Path(config.ORCHESTRATOR_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "orchestrator_decisions.jsonl"

        # Restore state from log
        self._restore_state()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, is_running: Callable[[], bool]) -> None:
        """Main loop: tick every ORCHESTRATOR_INTERVAL_SECONDS."""
        # Small initial delay to let the engine warm up
        await asyncio.sleep(15)

        while is_running():
            try:
                decision = await self._tick()
                if decision:
                    self._log(decision)
                    logger.info(
                        "Orchestrator [%s] → %s: %s",
                        decision.state, decision.action, decision.reason,
                    )
            except Exception:
                logger.exception("Orchestrator tick failed")

            # Interruptible sleep
            remaining = float(self._interval)
            while remaining > 0 and is_running():
                step = min(5.0, remaining)
                await asyncio.sleep(step)
                remaining -= step

    async def _tick(self) -> Optional[OrchestratorDecision]:
        """Dispatch based on current state."""
        if self._state == COLLECTING:
            return await self._handle_collecting()
        elif self._state == TRAINING:
            return await self._handle_training()
        elif self._state == ACTIVATING:
            return await self._handle_activating()
        elif self._state == MONITORING:
            return await self._handle_monitoring()
        return None

    # ------------------------------------------------------------------
    # COLLECTING: wait for sufficient training data
    # ------------------------------------------------------------------

    async def _handle_collecting(self) -> OrchestratorDecision:
        row_counts = self._count_data_rows()
        total_rows = sum(row_counts.values())
        symbols_ready = sum(
            1 for count in row_counts.values()
            if count >= config.ORCHESTRATOR_MIN_DATA_ROWS
        )

        metrics = {
            "total_rows": total_rows,
            "symbols_with_data": len(row_counts),
            "symbols_ready": symbols_ready,
            "min_required": config.ORCHESTRATOR_MIN_DATA_ROWS,
        }

        if symbols_ready >= _MIN_SYMBOLS_WITH_DATA:
            self._state = TRAINING
            return OrchestratorDecision(
                ts=datetime.now(timezone.utc).isoformat(),
                state=COLLECTING,
                action="transition_to_training",
                reason=f"{symbols_ready} symbols have >= {config.ORCHESTRATOR_MIN_DATA_ROWS} rows",
                metrics=metrics,
            )

        return OrchestratorDecision(
            ts=datetime.now(timezone.utc).isoformat(),
            state=COLLECTING,
            action="wait",
            reason=f"Only {symbols_ready}/{_MIN_SYMBOLS_WITH_DATA} symbols ready ({total_rows} total rows)",
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # TRAINING: run model comparison + selection
    # ------------------------------------------------------------------

    async def _handle_training(self) -> OrchestratorDecision:
        try:
            result = await asyncio.to_thread(self._run_full_training)
        except Exception as e:
            self._consecutive_train_failures += 1
            self._last_training_error = f"training_failed: {e}"
            logger.exception("Training failed")

            if self._consecutive_train_failures >= 3:
                self._state = COLLECTING
                return OrchestratorDecision(
                    ts=datetime.now(timezone.utc).isoformat(),
                    state=TRAINING,
                    action="back_to_collecting",
                    reason=f"3 consecutive training failures: {e}",
                )

            self._state = COLLECTING
            return OrchestratorDecision(
                ts=datetime.now(timezone.utc).isoformat(),
                state=TRAINING,
                action="training_failed",
                reason=str(e),
                details={"consecutive_failures": self._consecutive_train_failures},
            )

        contrarian_auc = result.get("contrarian_auc", 0.0)
        regime_accuracy = result.get("regime_accuracy", 0.0)

        metrics = {
            "contrarian_auc": contrarian_auc,
            "contrarian_selected": result.get("contrarian_selected", ""),
            "regime_accuracy": regime_accuracy,
            "regime_selected": result.get("regime_selected", ""),
        }

        min_auc = 0.60
        min_regime_accuracy = 0.50  # allow regime-only activation when contrarian unavailable
        contrarian_ok = contrarian_auc >= min_auc
        regime_ok = regime_accuracy >= min_regime_accuracy

        if contrarian_ok:
            self._last_train_auc = contrarian_auc
            self._last_train_ts = datetime.now(timezone.utc)
            self._train_count += 1
            self._consecutive_train_failures = 0
            self._last_training_error = None
            self._active_contrarian_model = result.get("contrarian_selected", "xgboost")
            self._active_regime_model = result.get("regime_selected")
            self._state = ACTIVATING
            return OrchestratorDecision(
                ts=datetime.now(timezone.utc).isoformat(),
                state=TRAINING,
                action="transition_to_activating",
                reason=f"Contrarian AUC={contrarian_auc:.3f} >= {min_auc}",
                metrics=metrics,
            )

        # Contrarian failed or low AUC — still activate if regime trained (e.g. XGBoost/libomp missing)
        if regime_ok:
            self._last_train_auc = contrarian_auc
            self._last_train_ts = datetime.now(timezone.utc)
            self._train_count += 1
            self._consecutive_train_failures = 0
            self._last_training_error = (
                f"Contrarian AUC {contrarian_auc:.3f} < {min_auc} (e.g. libomp missing); regime-only."
            )
            self._active_contrarian_model = result.get("contrarian_selected") or None
            self._active_regime_model = result.get("regime_selected")
            self._state = ACTIVATING
            return OrchestratorDecision(
                ts=datetime.now(timezone.utc).isoformat(),
                state=TRAINING,
                action="transition_to_activating",
                reason=f"Regime accuracy={regime_accuracy:.3f} >= {min_regime_accuracy}; contrarian skipped",
                metrics=metrics,
            )

        # Both contrarian and regime insufficient — back to collecting
        self._consecutive_train_failures += 1
        self._last_training_error = f"Contrarian AUC {contrarian_auc:.3f} < {min_auc}; regime {regime_accuracy:.3f} < {min_regime_accuracy}"
        self._state = COLLECTING
        return OrchestratorDecision(
            ts=datetime.now(timezone.utc).isoformat(),
            state=TRAINING,
            action="back_to_collecting",
            reason=f"Contrarian AUC={contrarian_auc:.3f} < {min_auc}",
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # ACTIVATING: enable strategies on the live engine
    # ------------------------------------------------------------------

    async def _handle_activating(self) -> OrchestratorDecision:
        details: Dict[str, Any] = {}

        try:
            self._engine.enable_contrarian()
            details["contrarian"] = "enabled"
        except Exception as e:
            details["contrarian"] = f"failed: {e}"
            logger.exception("Failed to enable contrarian strategy")

        try:
            self._engine.enable_regime()
            details["regime"] = "enabled"
        except Exception as e:
            details["regime"] = f"failed: {e}"
            logger.exception("Failed to enable regime adapter")

        self._state = MONITORING
        return OrchestratorDecision(
            ts=datetime.now(timezone.utc).isoformat(),
            state=ACTIVATING,
            action="transition_to_monitoring",
            reason="Strategies activated",
            details=details,
        )

    # ------------------------------------------------------------------
    # MONITORING: evaluate performance
    # ------------------------------------------------------------------

    async def _handle_monitoring(self) -> OrchestratorDecision:
        metrics = self._collect_monitoring_metrics()

        mode = "rules"
        decision = self._rules_monitoring_decision(metrics)
        action = decision.get("action", "continue")
        reason = decision.get("reason", "")
        adjustments = decision.get("adjustments", [])

        # Execute the action
        details: Dict[str, Any] = {"mode": mode, "adjustments": adjustments}

        if action == "retrain":
            self._state = TRAINING
            details["transition"] = "TRAINING"

        elif action == "disable":
            config.CONTRARIAN_ENABLED = False
            config.REGIME_ENABLED = False
            self._state = COLLECTING
            details["transition"] = "COLLECTING"

        elif action == "adjust":
            for adj in adjustments:
                key = adj.get("key", "")
                value = adj.get("value")
                if key and value is not None and hasattr(config, key):
                    try:
                        current = getattr(config, key)
                        if isinstance(current, float):
                            setattr(config, key, float(value))
                        elif isinstance(current, int):
                            setattr(config, key, int(value))
                        elif isinstance(current, bool):
                            setattr(config, key, bool(value))
                        details.setdefault("applied_adjustments", []).append(adj)
                    except (ValueError, TypeError):
                        pass

        return OrchestratorDecision(
            ts=datetime.now(timezone.utc).isoformat(),
            state=MONITORING,
            action=action,
            reason=reason,
            metrics=metrics,
            details=details,
        )

    # ------------------------------------------------------------------
    # Monitoring metrics collection
    # ------------------------------------------------------------------

    def _collect_monitoring_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "model_auc": self._last_train_auc,
            "train_count": self._train_count,
        }

        # Model age
        if self._last_train_ts:
            age_hours = (datetime.now(timezone.utc) - self._last_train_ts).total_seconds() / 3600
            metrics["model_age_hours"] = round(age_hours, 1)
        else:
            metrics["model_age_hours"] = None

        # Contrarian performance from engine state
        try:
            state = self._engine.get_state()
            contrarian = state.get("contrarian", {})
            metrics["contrarian_trade_count"] = contrarian.get("trade_count", 0)
            metrics["contrarian_win_rate"] = contrarian.get("win_rate")
            metrics["contrarian_signal_count"] = contrarian.get("signal_count", 0)
            metrics["contrarian_model"] = contrarian.get("model")

            learner = state.get("contrarian_learner")
            if learner:
                metrics["learner_auc"] = learner.get("current_auc", 0)
                metrics["learner_accuracy"] = learner.get("prediction_accuracy", 0)
                metrics["learner_total_predictions"] = learner.get("prediction_total", 0)

            regime = state.get("regime")
            if regime:
                metrics["regime_state"] = regime.get("regime", None)
                metrics["regime_label"] = regime.get("regime_label", "unknown")
        except Exception:
            logger.debug("Could not collect engine state for monitoring")

        return metrics

    # ------------------------------------------------------------------
    # Deterministic rules
    # ------------------------------------------------------------------

    def _rules_monitoring_decision(self, metrics: Dict[str, Any]) -> dict:
        model_age_hours = metrics.get("model_age_hours")
        win_rate = metrics.get("contrarian_win_rate")
        trade_count = metrics.get("contrarian_trade_count", 0)
        min_trades = config.ORCHESTRATOR_MIN_TRADES_FOR_EVAL

        # Rule: disable after 3 consecutive train failures
        if self._consecutive_train_failures >= 3:
            return {
                "action": "disable",
                "reason": f"{self._consecutive_train_failures} consecutive training failures",
                "adjustments": [],
            }

        # Rule: retrain if model is too old
        max_age = config.ORCHESTRATOR_MAX_MODEL_AGE_HOURS
        if model_age_hours is not None and model_age_hours > max_age:
            return {
                "action": "retrain",
                "reason": f"Model age {model_age_hours:.0f}h exceeds {max_age}h limit",
                "adjustments": [],
            }

        # Rule: retrain if win rate too low with enough trades
        if (
            win_rate is not None
            and trade_count >= min_trades
            and win_rate < config.ORCHESTRATOR_MIN_WIN_RATE
        ):
            return {
                "action": "retrain",
                "reason": f"Win rate {win_rate:.1%} < {config.ORCHESTRATOR_MIN_WIN_RATE:.0%} over {trade_count} trades",
                "adjustments": [],
            }

        # Rule: check AUC regression from learner
        learner_auc = metrics.get("learner_auc", 0)
        if learner_auc > 0 and self._last_train_auc > 0:
            auc_drop = self._last_train_auc - learner_auc
            if auc_drop > 0.05:
                return {
                    "action": "retrain",
                    "reason": f"AUC dropped {auc_drop:.3f} (from {self._last_train_auc:.3f} to {learner_auc:.3f})",
                    "adjustments": [],
                }

        return {
            "action": "continue",
            "reason": "All metrics within acceptable bounds",
            "adjustments": [],
        }

    # ------------------------------------------------------------------
    # Data row counting
    # ------------------------------------------------------------------

    def _count_data_rows(self) -> Dict[str, int]:
        """Count CSV rows per symbol in data/funding_history/funding_rates/."""
        rate_dir = Path("data/funding_history/funding_rates")
        counts: Dict[str, int] = {}
        if not rate_dir.exists():
            return counts

        for csv_path in rate_dir.glob("*.csv"):
            symbol = csv_path.stem
            try:
                # Count lines minus header
                with csv_path.open("r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f) - 1
                if line_count > 0:
                    counts[symbol] = line_count
            except Exception:
                pass
        return counts

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def _run_full_training(self) -> dict:
        """Synchronous: build features, train models, return results.
        Called via asyncio.to_thread to avoid blocking the event loop.
        """
        result: Dict[str, Any] = {}

        # --- Contrarian model training ---
        try:
            from funding.ml.contrarian_features import build_contrarian_features_all
            from funding.ml.model_selector import ModelSelector

            # Discover symbols from CSV files
            rate_dir = Path("data/funding_history/funding_rates")
            symbols = [
                p.stem for p in rate_dir.glob("*.csv")
            ] if rate_dir.exists() else []

            if not symbols:
                raise RuntimeError("No funding rate CSV files found")

            df = build_contrarian_features_all(symbols)
            if df is None or len(df) < config.ORCHESTRATOR_MIN_DATA_ROWS:
                raise RuntimeError(
                    f"Insufficient feature rows: {len(df) if df is not None else 0}"
                )

            selector = ModelSelector()
            comparison = selector.compare(df, n_trials=20)
            selected = selector.select_best()
            model_metrics = comparison.get(selected, {})

            result["contrarian_selected"] = selected
            result["contrarian_auc"] = float(model_metrics.get("auc", 0))
            result["contrarian_accuracy"] = float(model_metrics.get("direction_accuracy", 0))
            result["contrarian_sharpe"] = float(model_metrics.get("sharpe_simulated", 0))

            logger.info(
                "Contrarian training complete: selected=%s AUC=%.3f",
                selected, result["contrarian_auc"],
            )
        except Exception as e:
            logger.exception("Contrarian model training failed")
            result["contrarian_error"] = str(e)
            result["contrarian_auc"] = 0.0

        # --- Regime model training ---
        try:
            from funding.ml.regime_features import build_regime_features
            from funding.ml.regime_selector import RegimeSelector

            # Use same symbols
            rate_dir = Path("data/funding_history/funding_rates")
            symbols = [p.stem for p in rate_dir.glob("*.csv")] if rate_dir.exists() else []

            if symbols:
                regime_df = build_regime_features(symbols)
                if regime_df is not None and not regime_df.empty:
                    regime_selector = RegimeSelector()
                    regime_comparison = regime_selector.compare(regime_df)
                    regime_selected = regime_selector.select_best()
                    regime_metrics = regime_comparison.get(regime_selected, {})

                    result["regime_selected"] = regime_selected
                    result["regime_accuracy"] = float(regime_metrics.get("accuracy", 0))
                    result["regime_stability"] = float(regime_metrics.get("stability", 0))

                    logger.info(
                        "Regime training complete: selected=%s accuracy=%.3f",
                        regime_selected, result["regime_accuracy"],
                    )
                else:
                    result["regime_error"] = "Empty regime features"
        except Exception as e:
            logger.warning("Regime model training failed (non-critical): %s", e)
            result["regime_error"] = str(e)

        return result

    # ------------------------------------------------------------------
    # State persistence / restoration
    # ------------------------------------------------------------------

    def _restore_state(self) -> None:
        """Read last line of JSONL log to resume state after restart."""
        if not self._log_file.exists():
            return

        last_line = None
        try:
            with self._log_file.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                data = b""
                block = min(8192, size)
                if block > 0:
                    f.seek(size - block)
                    data = f.read(block)
            lines = data.decode("utf-8", "ignore").strip().splitlines()
            if lines:
                last_line = json.loads(lines[-1])
        except Exception:
            return

        if not last_line:
            return

        prev_state = last_line.get("state", "")
        prev_action = last_line.get("action", "")

        # If we were in MONITORING or just transitioned to it, resume there
        # (models should still be on disk)
        if prev_state == MONITORING or prev_action == "transition_to_monitoring":
            # Check if models still exist on disk
            try:
                from funding.ml.model_selector import ModelSelector
                s = ModelSelector()
                s.load_comparison()
                s.get_model()
                self._state = MONITORING

                # Restore metrics from log
                metrics = last_line.get("metrics", {})
                self._last_train_auc = float(metrics.get("model_auc", 0))
                ts_str = last_line.get("ts", "")
                if ts_str:
                    try:
                        self._last_train_ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        pass

                logger.info("Orchestrator restored to MONITORING state from log")
                return
            except Exception:
                pass

        # If we were training or activating, restart from COLLECTING
        # to avoid partial state
        if prev_state in (TRAINING, ACTIVATING):
            self._state = COLLECTING
            logger.info("Orchestrator restored to COLLECTING (was %s)", prev_state)
            return

        # Default: stay in COLLECTING
        logger.info("Orchestrator starting in COLLECTING state")

    def _log(self, decision: OrchestratorDecision) -> None:
        """Append decision to JSONL."""
        payload = {
            "ts": decision.ts,
            "state": decision.state,
            "action": decision.action,
            "reason": decision.reason,
            "metrics": decision.metrics,
            "details": decision.details,
        }
        try:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("Failed to write orchestrator decision log")

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """For dashboard API: state, metrics, last decision, model info."""
        data_rows = self._count_data_rows()
        return {
            "state": self._state,
            "interval_seconds": self._interval,
            "train_count": self._train_count,
            "last_train_auc": self._last_train_auc,
            "last_train_ts": self._last_train_ts.isoformat() if self._last_train_ts else None,
            "model_age_hours": (
                round(
                    (datetime.now(timezone.utc) - self._last_train_ts).total_seconds() / 3600, 1
                )
                if self._last_train_ts
                else None
            ),
            "consecutive_train_failures": self._consecutive_train_failures,
            "last_training_error": self._last_training_error,
            "active_contrarian_model": self._active_contrarian_model,
            "active_regime_model": self._active_regime_model,
            "data_rows": data_rows,
            "total_data_rows": sum(data_rows.values()),
            "min_data_rows_required": config.ORCHESTRATOR_MIN_DATA_ROWS,
        }
