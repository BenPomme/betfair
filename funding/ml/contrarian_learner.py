"""
Online learning loop for the contrarian directional model.

Analogous to funding.ml.online_learner (FundingOnlineLearner) but targets
the contrarian price-direction model instead of the funding-rate predictor.

Runs as an asyncio task inside FundingEngine:
  - Every FUNDING_RETRAIN_INTERVAL_HOURS: retrains via ModelSelector.compare()
  - Gate check: new model must have AUC >= 0.60 and not regress > 0.05 from
    the previous best
  - On acceptance: saves the model and updates ContrarianStrategy's live model
  - Tracks prediction accuracy by comparing signal direction to actual outcomes
    recorded via log_trade_outcome()
"""
import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import config
from funding.ml.contrarian_features import load_quarantined_symbols
from funding.ml.learning_quality import FundingLearningQuality

logger = logging.getLogger(__name__)

# Minimum AUC gate for accepting a newly retrained contrarian model.
# Deliberately lower than the funding-rate predictor gate (0.75) because
# direction prediction is inherently harder than rate-magnitude prediction.
_MIN_AUC = 0.60

# Maximum tolerated AUC regression vs. the incumbent model before rejection.
_MAX_AUC_REGRESSION = 0.05

# Minimum number of feature rows required to attempt a retrain.
_MIN_ROWS = 500

class ContrarianOnlineLearner:
    """Continuously retrains the contrarian model as new market data arrives.

    Parameters
    ----------
    watchlist_fn:
        Callable returning a set/list of active symbol strings.
        Defaults to an empty-set lambda when not supplied.
    model_selector:
        A funding.ml.model_selector.ModelSelector instance (or duck-type
        equivalent) with a ``compare(df)`` method.  When None the learner
        will instantiate one lazily on first retrain.
    contrarian_strategy:
        The live funding.strategy.contrarian_strategy.ContrarianStrategy
        instance whose ``._model`` attribute should be updated after a
        successful retrain.  When None the strategy reference is not updated
        (useful for unit tests).
    """

    def __init__(
        self,
        watchlist_fn: Optional[Callable[[], Any]] = None,
        model_selector: Optional[Any] = None,
        contrarian_strategy: Optional[Any] = None,
        trade_log_path: Optional[str] = None,
        quality_state_path: Optional[str] = None,
    ) -> None:
        self._watchlist_fn = watchlist_fn or (lambda: set())
        self._model_selector = model_selector
        self._contrarian_strategy = contrarian_strategy

        self._running: bool = False
        self._trade_log_path = Path(trade_log_path or "data/funding_models/contrarian_trade_log.jsonl")

        # Retrain tracking
        self._last_retrain_ts: float = 0.0
        self._last_retrain: Optional[str] = None   # ISO timestamp string
        self._retrain_count: int = 0
        self._last_retrain_result: Optional[str] = None   # "accepted" / "rejected"
        self._current_auc: float = 0.0
        self._retrain_history: List[Dict[str, Any]] = []

        # Prediction accuracy tracking (populated by log_trade_outcome)
        self._accuracy_correct: int = 0
        self._accuracy_total: int = 0

        # Internal state dict (returned by get_state)
        self._state: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []
        self._total_realized_pnl: float = 0.0
        self._quality = FundingLearningQuality(
            model_id="contrarian_online_learner",
            model_family="contrarian",
            state_path=quality_state_path or "data/funding/state/contrarian_online_learner_quality.json",
        )

        # Load last-known AUC from comparison file if present
        self._load_incumbent_auc()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: trigger periodic retrains until stop() is called."""
        self._running = True
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ContrarianOnlineLearner started "
            "(retrain_interval=%dh, min_auc=%.2f, max_regression=%.2f)",
            config.FUNDING_RETRAIN_INTERVAL_HOURS,
            _MIN_AUC,
            _MAX_AUC_REGRESSION,
        )

        while self._running:
            try:
                elapsed_hours = (time.monotonic() - self._last_retrain_ts) / 3600
                if (
                    self._last_retrain_ts == 0.0
                    or elapsed_hours >= config.FUNDING_RETRAIN_INTERVAL_HOURS
                ):
                    await self._try_retrain()
            except Exception as exc:
                logger.exception("ContrarianOnlineLearner error: %s", exc)

            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._running = False

    def drain_events(self) -> List[Dict[str, Any]]:
        events = list(self._events)
        self._events.clear()
        return events

    # ------------------------------------------------------------------
    # Trade outcome logging
    # ------------------------------------------------------------------

    def log_trade_outcome(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
    ) -> None:
        """Append a closed trade outcome to the contrarian trade log.

        Called by the engine after a directional position is closed so that
        prediction accuracy can be computed across the rolling retrain window.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. "ETHUSDT").
        direction:
            Signal direction at entry ("LONG" or "SHORT").
        entry_price:
            Execution price at entry.
        exit_price:
            Execution price at exit.
        """
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)

        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
        # For a LONG position: pnl_pct > 0 is correct; SHORT: pnl_pct < 0 is correct.
        if direction.upper() == "LONG":
            correct = pnl_pct > 0.0
        else:  # SHORT
            correct = pnl_pct < 0.0

        self._accuracy_total += 1
        if correct:
            self._accuracy_correct += 1
        self._total_realized_pnl += float(pnl_pct)
        pred_prob = 0.5 + min(0.49, abs(float(pnl_pct)) * 5.0)
        label = 1 if pnl_pct > 0 else 0
        base_prob = 0.5
        gate_events = self._quality.record_settlement(
            symbol=symbol,
            stake=1.0,
            pnl=float(pnl_pct),
            label=label,
            pred_prob=pred_prob,
            base_prob=base_prob,
            clv=None,
        )
        self._events.extend(gate_events)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry_price": round(entry_price, 8),
            "exit_price": round(exit_price, 8),
            "pnl_pct": round(pnl_pct, 6),
            "correct": correct,
        }
        try:
            with open(self._trade_log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Failed to write contrarian trade log entry: %s", exc)

    def record_signal_probability(self, confidence: float) -> None:
        try:
            c = float(confidence)
        except Exception:
            return
        c = max(0.0, min(1.0, c))
        self._quality.add_prediction(c)
        if self._quality.prediction_is_frozen():
            self._events.append(
                {
                    "kind": "funding_prediction_frozen",
                    "model_id": "contrarian_online_learner",
                    "window": int(config.FUNDING_FROZEN_WINDOW),
                }
            )
        sat = self._quality.saturation_rate()
        if sat >= float(config.FUNDING_SATURATION_RATE_THRESHOLD):
            self._events.append(
                {
                    "kind": "funding_prediction_saturation",
                    "model_id": "contrarian_online_learner",
                    "window": int(config.FUNDING_SATURATION_WINDOW),
                    "rate": round(sat, 4),
                }
            )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return the current learner state for the dashboard.

        Returns
        -------
        dict with keys:
            running, current_auc, retrain_count, last_retrain_time,
            last_retrain_result, next_retrain_hours, retrain_interval_hours,
            min_auc_threshold, prediction_accuracy, prediction_total,
            retrain_history (last 10 entries).
        """
        accuracy = (
            self._accuracy_correct / self._accuracy_total
            if self._accuracy_total > 0
            else 0.0
        )

        if self._last_retrain_ts > 0:
            elapsed_h = (time.monotonic() - self._last_retrain_ts) / 3600
            next_retrain_h = max(0.0, config.FUNDING_RETRAIN_INTERVAL_HOURS - elapsed_h)
        else:
            next_retrain_h = 0.0

        gate_mult, gate_edge_bump = self._quality.gate_policy()
        quality_state = self._quality.state()
        return {
            "running": self._running,
            "current_auc": round(self._current_auc, 4),
            "retrain_count": self._retrain_count,
            "last_retrain_time": self._last_retrain,
            "last_retrain_result": self._last_retrain_result,
            "next_retrain_hours": round(next_retrain_h, 1),
            "retrain_interval_hours": config.FUNDING_RETRAIN_INTERVAL_HOURS,
            "min_auc_threshold": _MIN_AUC,
            "prediction_accuracy": round(accuracy, 4),
            "prediction_total": self._accuracy_total,
            "total_realized_pnl": round(self._total_realized_pnl, 6),
            "gate_multiplier": round(gate_mult, 4),
            "gate_edge_bump": round(gate_edge_bump, 6),
            "retrain_history": self._retrain_history[-10:],
            **quality_state,
        }

    # ------------------------------------------------------------------
    # Internal: retrain pipeline
    # ------------------------------------------------------------------

    async def _try_retrain(self) -> None:
        """Attempt to retrain the contrarian model on fresh data.

        Steps:
          1. Resolve watchlist symbols.
          2. Build contrarian feature matrix via build_contrarian_features_all.
          3. Check minimum row count (500 rows).
          4. Run ModelSelector.compare(df) in a thread (CPU-bound).
          5. Apply quality gates: AUC >= 0.60, regression <= 0.05.
          6. Accept: save the winning model; update ContrarianStrategy reference.
          7. Reject: keep old model, log rejection reason.
        """
        symbols = self._get_watchlist()
        now_str = datetime.now(timezone.utc).isoformat()
        if not symbols:
            logger.debug("ContrarianOnlineLearner: empty watchlist, skipping retrain")
            self._record_retrain(
                {
                    "time": now_str,
                    "selected_model": "n/a",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "sharpe": 0.0,
                    "data_rows": 0,
                    "result": "skipped",
                    "reason": "empty_watchlist",
                }
            )
            return

        logger.info(
            "ContrarianOnlineLearner: starting retrain with %d symbols", len(symbols)
        )

        # --- Step 1: Build feature matrix ---
        try:
            from funding.ml.contrarian_features import build_contrarian_features_all

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, build_contrarian_features_all, list(symbols)
            )
        except Exception as exc:
            logger.warning(
                "ContrarianOnlineLearner: feature building failed: %s", exc
            )
            self._record_retrain(
                {
                    "time": now_str,
                    "selected_model": "n/a",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "sharpe": 0.0,
                    "data_rows": 0,
                    "result": "rejected",
                    "reason": f"feature_build_failed: {exc}",
                }
            )
            return

        if df is None or df.empty or len(df) < _MIN_ROWS:
            logger.info(
                "ContrarianOnlineLearner: insufficient data (%d rows, need %d) — "
                "skipping retrain",
                len(df) if df is not None and not df.empty else 0,
                _MIN_ROWS,
            )
            self._record_retrain(
                {
                    "time": now_str,
                    "selected_model": "n/a",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "sharpe": 0.0,
                    "data_rows": int(len(df) if df is not None else 0),
                    "result": "skipped",
                    "reason": "insufficient_rows",
                }
            )
            return

        reject = self._validate_training_df(df)
        if reject is not None:
            self._events.append(reject)
            self._record_retrain(
                {
                    "time": now_str,
                    "selected_model": "n/a",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "sharpe": 0.0,
                    "data_rows": int(len(df)),
                    "result": "rejected",
                    "reason": str(reject.get("reason", "validation_failed")),
                }
            )
            return

        # --- Step 2: Train / compare models ---
        try:
            selector = self._get_or_create_selector()
        except Exception as exc:
            logger.warning(
                "ContrarianOnlineLearner: could not create ModelSelector (e.g. XGBoost/libomp missing): %s",
                exc,
            )
            self._record_retrain(
                {
                    "time": now_str,
                    "selected_model": "n/a",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "sharpe": 0.0,
                    "data_rows": int(len(df)),
                    "result": "rejected",
                    "reason": f"selector_init_failed: {exc}",
                }
            )
            return
        try:
            loop = asyncio.get_event_loop()
            comparison = await loop.run_in_executor(
                None,
                lambda: selector.compare(
                    df,
                    n_trials=max(1, int(getattr(config, "CONTRARIAN_RETRAIN_XGB_TRIALS", 3))),
                    tft_epochs=max(1, int(getattr(config, "CONTRARIAN_RETRAIN_TFT_EPOCHS", 5))),
                ),
            )
        except Exception as exc:
            logger.warning(
                "ContrarianOnlineLearner: ModelSelector.compare failed: %s", exc
            )
            self._record_retrain(
                {
                    "time": now_str,
                    "selected_model": "n/a",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "sharpe": 0.0,
                    "data_rows": int(len(df)),
                    "result": "rejected",
                    "reason": f"compare_failed: {exc}",
                }
            )
            return

        # --- Step 3: Extract best-model metrics ---
        selected = comparison.get("selected", "xgboost")
        model_entry = comparison.get(selected, {})
        new_auc = float(model_entry.get("auc", 0.0))
        new_sharpe = float(model_entry.get("sharpe_simulated", 0.0))

        # --- Step 4: Quality gates ---
        auc_meets_min = new_auc >= _MIN_AUC
        auc_no_regression = new_auc >= self._current_auc - _MAX_AUC_REGRESSION

        retrain_record: Dict[str, Any] = {
            "time": now_str,
            "selected_model": selected,
            "new_auc": round(new_auc, 4),
            "old_auc": round(self._current_auc, 4),
            "sharpe": round(new_sharpe, 4),
            "data_rows": len(df),
        }

        if auc_meets_min and auc_no_regression:
            # --- Step 5a: Accept ---
            try:
                new_model = selector.get_model()
                if new_model is None:
                    logger.warning(
                        "ContrarianOnlineLearner: select_best returned no model (XGBoost+TFT unavailable)."
                    )
                    retrain_record["result"] = "rejected"
                    retrain_record["reason"] = "no model available (install libomp for XGBoost or use TFT)"
                    self._last_retrain_result = "rejected"
                    self._record_retrain(retrain_record)
                    return
                # Persist the winning model to disk
                if hasattr(new_model, "save"):
                    await loop.run_in_executor(None, new_model.save)

                # Live-swap the model inside ContrarianStrategy
                if self._contrarian_strategy is not None:
                    self._contrarian_strategy._model = new_model
                    logger.info(
                        "ContrarianOnlineLearner: live model swapped to %s",
                        type(new_model).__name__,
                    )
            except Exception as exc:
                logger.warning(
                    "ContrarianOnlineLearner: model save/swap failed: %s", exc
                )

            old_auc = self._current_auc
            self._current_auc = new_auc
            retrain_record["result"] = "accepted"
            self._last_retrain_result = "accepted"
            logger.info(
                "ContrarianOnlineLearner: retrain ACCEPTED — "
                "model=%s, AUC %.4f → %.4f, Sharpe %.4f (%d rows)",
                selected,
                old_auc,
                new_auc,
                new_sharpe,
                len(df),
            )
        else:
            # --- Step 5b: Reject ---
            reasons: List[str] = []
            if not auc_meets_min:
                reasons.append(
                    f"AUC {new_auc:.4f} < min {_MIN_AUC}"
                )
            if not auc_no_regression:
                reasons.append(
                    f"AUC {new_auc:.4f} regressed from {self._current_auc:.4f} "
                    f"(max regression {_MAX_AUC_REGRESSION})"
                )
            reason_str = "; ".join(reasons)
            retrain_record["result"] = "rejected"
            retrain_record["reason"] = reason_str
            self._last_retrain_result = "rejected"
            logger.info(
                "ContrarianOnlineLearner: retrain REJECTED — %s", reason_str
            )

        # --- Step 6: Record ---
        self._record_retrain(retrain_record)

    def _validate_training_df(self, df) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        sample_features: Dict[str, float] = {}
        for col in df.columns:
            if str(col) in {"target", "label", "target_direction"}:
                continue
            try:
                series = (
                    df[col]
                    .replace([float("inf"), float("-inf")], float("nan"))
                    .dropna()
                )
                if len(series) == 0:
                    continue
                sample_features[str(col)] = float(series.iloc[-1])
            except Exception:
                continue
        if not sample_features:
            return None
        reject = self._quality.validate_features(
            sample_features,
            symbol="*",
            context="contrarian_retrain_data",
        )
        if reject is not None:
            return reject
        self._events.extend(self._quality.update_feature_drift(sample_features, symbol="*"))
        label_cols = [
            c
            for c in df.columns
            if str(c) in {
                "target",
                "label",
                "target_direction",
                "price_return_24h_target",
                "direction_24h",
            }
        ]
        for col in label_cols:
            for val in df[col].tail(200).values.tolist():
                if str(col) == "price_return_24h_target":
                    try:
                        if not math.isfinite(float(val)):
                            return {
                                "kind": "funding_update_rejected",
                                "model_id": "contrarian_online_learner",
                                "symbol": "*",
                                "context": "contrarian_retrain_data",
                                "reason": "non_finite_regression_target",
                                "label": str(val),
                            }
                    except Exception:
                        return {
                            "kind": "funding_update_rejected",
                            "model_id": "contrarian_online_learner",
                            "symbol": "*",
                            "context": "contrarian_retrain_data",
                            "reason": "non_numeric_regression_target",
                            "label": str(val),
                        }
                    continue
                if val not in (0, 1):
                    return {
                        "kind": "funding_update_rejected",
                        "model_id": "contrarian_online_learner",
                        "symbol": "*",
                        "context": "contrarian_retrain_data",
                        "reason": "invalid_label",
                        "label": str(val),
                    }
        return None

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _get_watchlist(self) -> List[str]:
        """Resolve the current watchlist symbols.

        Falls back to CSV discovery in data/funding_history/funding_rates/
        when the watchlist callable returns nothing.
        """
        try:
            result = self._watchlist_fn()
            if result:
                max_symbols = max(1, int(getattr(config, "FUNDING_CONTRARIAN_RETRAIN_MAX_SYMBOLS", 20)))
                quarantined = load_quarantined_symbols()
                filtered = [sym for sym in list(result) if str(sym).upper() not in quarantined]
                return filtered[:max_symbols]
        except Exception:
            pass

        # Fallback: discover from persisted funding-rate CSV files
        rates_dir = Path("data/funding_history/funding_rates")
        if rates_dir.exists():
            max_symbols = max(1, int(getattr(config, "FUNDING_CONTRARIAN_RETRAIN_MAX_SYMBOLS", 20)))
            quarantined = load_quarantined_symbols()
            discovered = [f.stem for f in rates_dir.glob("*.csv") if f.stem.upper() not in quarantined]
            return discovered[:max_symbols]
        return []

    def _get_or_create_selector(self) -> Any:
        """Return the ModelSelector, creating a default one if needed."""
        if self._model_selector is None:
            from funding.ml.model_selector import ModelSelector

            self._model_selector = ModelSelector()
            logger.debug(
                "ContrarianOnlineLearner: created default ModelSelector"
            )
        return self._model_selector

    def _load_incumbent_auc(self) -> None:
        """Populate self._current_auc from the last saved comparison file."""
        comparison_path = Path("data/funding_models/contrarian_comparison.json")
        if not comparison_path.exists():
            return
        try:
            data = json.loads(comparison_path.read_text())
            selected = data.get("selected", "xgboost")
            entry = data.get(selected, {})
            auc = float(entry.get("auc", 0.0))
            if auc > 0.0:
                self._current_auc = auc
                logger.info(
                    "ContrarianOnlineLearner: loaded incumbent AUC %.4f "
                    "(model=%s) from %s",
                    auc,
                    selected,
                    comparison_path,
                )
        except Exception as exc:
            logger.debug(
                "ContrarianOnlineLearner: could not load incumbent AUC: %s", exc
            )

    def _record_retrain(self, retrain_record: Dict[str, Any]) -> None:
        self._retrain_count += 1
        self._last_retrain_ts = time.monotonic()
        self._last_retrain = str(retrain_record.get("time"))
        self._last_retrain_result = str(retrain_record.get("result"))
        self._retrain_history.append(dict(retrain_record))
        if len(self._retrain_history) > 50:
            self._retrain_history = self._retrain_history[-50:]
