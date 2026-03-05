"""
Online learning loop for funding rate ML model.

Runs as an asyncio task inside FundingEngine:
  - After each funding settlement: logs prediction vs actual
  - Every FUNDING_RETRAIN_INTERVAL_HOURS: retrains model on updated data
  - Gate check: only deploys new model if AUC meets threshold
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from funding.data.funding_rate_fetcher import FundingRateFetcher
from funding.ml.feature_engineer import build_features_all_symbols
from funding.ml.funding_predictor import FundingPredictor

logger = logging.getLogger(__name__)

PREDICTION_LOG_PATH = Path("data/funding_models/prediction_log.jsonl")
SETTLEMENT_HOURS = {0, 8, 16}


class FundingOnlineLearner:
    """Continuously learns from funding rate outcomes."""

    def __init__(self, watchlist_fn=None):
        """
        Args:
            watchlist_fn: callable returning current watchlist symbols list.
        """
        self._watchlist_fn = watchlist_fn
        self._fetcher = FundingRateFetcher()

        # Retrain tracking
        self._last_retrain_ts: float = 0.0
        self._retrain_count: int = 0
        self._current_auc: float = 0.0
        self._last_retrain_result: Optional[str] = None  # "accepted" / "rejected"
        self._last_retrain_time: Optional[str] = None
        self._retrain_history: List[Dict[str, Any]] = []

        # Prediction tracking
        self._prediction_accuracy_correct: int = 0
        self._prediction_accuracy_total: int = 0

        # Settlement tracking
        self._last_settlement_hour: int = -1

        self._running = False

        # Load current model AUC if available
        meta_path = Path("data/funding_models/funding_predictor_meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                self._current_auc = meta.get("metrics", {}).get("direction_auc", 0.0)
            except Exception:
                pass

    async def run(self) -> None:
        """Main loop: check for settlements and retrain triggers."""
        self._running = True
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Online learner started (retrain_interval=%dh, min_auc=%.2f)",
                     config.FUNDING_RETRAIN_INTERVAL_HOURS, config.FUNDING_RETRAIN_MIN_AUC)

        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Check for settlement (backfill prediction log)
                if (now.hour in SETTLEMENT_HOURS
                        and now.hour != self._last_settlement_hour
                        and now.minute < 10):
                    self._last_settlement_hour = now.hour
                    await self._on_settlement()

                # Check for retrain trigger
                elapsed_hours = (time.monotonic() - self._last_retrain_ts) / 3600
                if (self._last_retrain_ts == 0.0
                        or elapsed_hours >= config.FUNDING_RETRAIN_INTERVAL_HOURS):
                    await self._try_retrain()

            except Exception as e:
                logger.exception("Online learner error: %s", e)

            await asyncio.sleep(60)

    def stop(self) -> None:
        self._running = False

    async def _on_settlement(self) -> None:
        """After each settlement, backfill actual rates into prediction log."""
        if not PREDICTION_LOG_PATH.exists():
            return

        # Read prediction log, find entries missing actual_rate
        updated_lines = []
        backfilled = 0
        try:
            with open(PREDICTION_LOG_PATH) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("actual_rate") is None and entry.get("symbol"):
                        # Try to fetch actual rate from API
                        actual = await self._fetch_actual_rate(entry["symbol"])
                        if actual is not None:
                            entry["actual_rate"] = actual
                            entry["actual_positive"] = actual > 0
                            pred_pos = entry.get("predicted_positive", None)
                            if pred_pos is not None:
                                correct = (pred_pos and actual > 0) or (not pred_pos and actual <= 0)
                                entry["correct"] = correct
                                self._prediction_accuracy_total += 1
                                if correct:
                                    self._prediction_accuracy_correct += 1
                            backfilled += 1
                    updated_lines.append(json.dumps(entry))
        except Exception as e:
            logger.warning("Error backfilling predictions: %s", e)
            return

        if backfilled > 0:
            with open(PREDICTION_LOG_PATH, "w") as f:
                f.write("\n".join(updated_lines) + "\n")
            logger.info("Backfilled %d prediction outcomes", backfilled)

    async def _fetch_actual_rate(self, symbol: str) -> Optional[float]:
        """Fetch the most recent settled funding rate for a symbol."""
        try:
            from binance.um_futures import UMFutures
            client = UMFutures(base_url="https://fapi.binance.com")
            result = client.funding_rate(symbol=symbol, limit=1)
            if result:
                return float(result[0]["fundingRate"])
        except Exception as e:
            logger.debug("Failed to fetch actual rate for %s: %s", symbol, e)
        return None

    async def _try_retrain(self) -> None:
        """Attempt to retrain the model on updated data."""
        symbols = self._get_watchlist()
        if not symbols:
            logger.debug("No watchlist symbols for retraining")
            self._last_retrain_ts = time.monotonic()
            return

        logger.info("Starting model retrain with %d symbols...", len(symbols))

        # Step 1: Fetch incremental data
        loop = asyncio.get_event_loop()
        update_counts = await loop.run_in_executor(
            None, self._fetcher.update_all, symbols
        )
        total_new = sum(update_counts.values())
        logger.info("Fetched %d new funding rate rows", total_new)

        # Step 2: Build feature matrix
        try:
            df = await loop.run_in_executor(
                None, build_features_all_symbols, symbols
            )
        except Exception as e:
            logger.warning("Feature building failed: %s", e)
            self._last_retrain_ts = time.monotonic()
            return

        if df.empty or len(df) < config.FUNDING_RETRAIN_MIN_ROWS:
            logger.info("Insufficient data for retrain: %d rows (need %d)",
                        len(df), config.FUNDING_RETRAIN_MIN_ROWS)
            self._last_retrain_ts = time.monotonic()
            return

        # Step 3: Train new model (no tuning for speed)
        try:
            predictor = FundingPredictor()
            metrics = await loop.run_in_executor(
                None, lambda: predictor.train(df, tune=False)
            )
        except Exception as e:
            logger.warning("Model training failed: %s", e)
            self._last_retrain_ts = time.monotonic()
            return

        new_auc = metrics.get("direction_auc", 0.0)
        now_str = datetime.now(timezone.utc).isoformat()

        # Step 4: Gate check
        auc_meets_min = new_auc >= config.FUNDING_RETRAIN_MIN_AUC
        auc_no_regression = new_auc >= self._current_auc - 0.02

        retrain_record = {
            "time": now_str,
            "new_auc": round(new_auc, 4),
            "old_auc": round(self._current_auc, 4),
            "data_rows": len(df),
            "new_rates_fetched": total_new,
        }

        if auc_meets_min and auc_no_regression:
            # Accept: save and reload
            await loop.run_in_executor(None, predictor.save)
            old_auc = self._current_auc
            self._current_auc = new_auc

            # Signal entry_strategy to reload
            from funding.strategy.entry_strategy import reload_ml_predictor
            reload_ml_predictor()

            retrain_record["result"] = "accepted"
            self._last_retrain_result = "accepted"
            logger.info(
                "Model retrain ACCEPTED: AUC %.4f → %.4f (%d rows)",
                old_auc, new_auc, len(df),
            )
        else:
            reasons = []
            if not auc_meets_min:
                reasons.append(f"AUC {new_auc:.4f} < min {config.FUNDING_RETRAIN_MIN_AUC}")
            if not auc_no_regression:
                reasons.append(f"AUC {new_auc:.4f} regressed from {self._current_auc:.4f}")
            reason_str = "; ".join(reasons)

            retrain_record["result"] = "rejected"
            retrain_record["reason"] = reason_str
            self._last_retrain_result = "rejected"
            logger.info("Model retrain REJECTED: %s", reason_str)

        self._retrain_count += 1
        self._last_retrain_ts = time.monotonic()
        self._last_retrain_time = now_str
        self._retrain_history.append(retrain_record)
        # Keep only last 50 records
        if len(self._retrain_history) > 50:
            self._retrain_history = self._retrain_history[-50:]

    def _get_watchlist(self) -> List[str]:
        """Get current watchlist symbols."""
        if self._watchlist_fn:
            try:
                return list(self._watchlist_fn())
            except Exception:
                pass
        # Fallback: discover from existing CSV files
        rates_dir = Path("data/funding_history/funding_rates")
        if rates_dir.exists():
            return [f.stem for f in rates_dir.glob("*.csv")]
        return []

    def get_state(self) -> Dict[str, Any]:
        """Get current learner state for dashboard."""
        accuracy = 0.0
        if self._prediction_accuracy_total > 0:
            accuracy = self._prediction_accuracy_correct / self._prediction_accuracy_total

        # Time until next retrain
        if self._last_retrain_ts > 0:
            elapsed_h = (time.monotonic() - self._last_retrain_ts) / 3600
            next_retrain_h = max(0, config.FUNDING_RETRAIN_INTERVAL_HOURS - elapsed_h)
        else:
            next_retrain_h = 0

        return {
            "running": self._running,
            "current_auc": round(self._current_auc, 4),
            "retrain_count": self._retrain_count,
            "last_retrain_time": self._last_retrain_time,
            "last_retrain_result": self._last_retrain_result,
            "next_retrain_hours": round(next_retrain_h, 1),
            "retrain_interval_hours": config.FUNDING_RETRAIN_INTERVAL_HOURS,
            "min_auc_threshold": config.FUNDING_RETRAIN_MIN_AUC,
            "prediction_accuracy": round(accuracy, 4),
            "prediction_total": self._prediction_accuracy_total,
            "retrain_history": self._retrain_history[-10:],
        }


def log_prediction(
    symbol: str,
    predicted_positive: bool,
    confidence: float,
    predicted_rate: float,
    current_rate: float,
    position_size: float,
) -> None:
    """Log a prediction entry for later backfilling with actuals."""
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "predicted_positive": predicted_positive,
        "confidence": round(confidence, 4),
        "predicted_rate": round(predicted_rate, 8),
        "current_rate": round(current_rate, 8),
        "position_size": round(position_size, 2),
        "actual_rate": None,
    }
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
