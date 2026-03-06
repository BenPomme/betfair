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
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from funding.data.funding_rate_fetcher import FundingRateFetcher
from funding.ml.feature_engineer import build_features_all_symbols
from funding.ml.funding_predictor import FundingPredictor
from funding.ml.learning_quality import FundingLearningQuality

logger = logging.getLogger(__name__)

PREDICTION_LOG_PATH = Path("data/funding_models/prediction_log.jsonl")
QUALITY_STATE_PATH = "data/funding/state/funding_online_learner_quality.json"
META_PATH = Path("data/funding_models/funding_predictor_meta.json")
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
        self._total_realized_pnl: float = 0.0

        # Settlement tracking
        self._last_settlement_hour: int = -1

        self._running = False
        self._events: List[Dict[str, Any]] = []
        self._pending_prediction_symbols: set = set()
        self._last_shadow_bucket: Optional[str] = None
        self._quality = FundingLearningQuality(
            model_id="funding_online_learner",
            model_family="funding",
            state_path=QUALITY_STATE_PATH,
        )

        # Load current model AUC if available
        meta_path = META_PATH
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                self._current_auc = meta.get("metrics", {}).get("direction_auc", 0.0)
            except Exception:
                pass
        self._load_pending_prediction_symbols()

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

                # Keep learning active even when no trades are opened:
                # periodically log one unresolved shadow prediction per symbol.
                bucket = now.strftime("%Y%m%d%H") + f":{0 if now.minute < 30 else 1}"
                if now.minute in {5, 35} and bucket != self._last_shadow_bucket:
                    self._last_shadow_bucket = bucket
                    await self._log_shadow_predictions(max_symbols=20)

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

    def drain_events(self) -> List[Dict[str, Any]]:
        events = list(self._events)
        self._events.clear()
        return events

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
                                stake = float(entry.get("position_size", 0.0))
                                pred_rate = float(entry.get("predicted_rate", actual))
                                curr_rate = float(entry.get("current_rate", 0.0))
                                pnl = stake * actual
                                self._total_realized_pnl += pnl
                                label = 1 if actual > 0 else 0
                                pred_prob = self._rate_to_prob(pred_rate)
                                base_prob = self._rate_to_prob(curr_rate)
                                gate_events = self._quality.record_settlement(
                                    symbol=entry["symbol"],
                                    stake=max(0.0, stake),
                                    pnl=pnl,
                                    label=label,
                                    pred_prob=pred_prob,
                                    base_prob=base_prob,
                                    clv=None,
                                )
                                self._events.extend(gate_events)
                            backfilled += 1
                            self._pending_prediction_symbols.discard(entry["symbol"])
                    updated_lines.append(json.dumps(entry))
        except Exception as e:
            logger.warning("Error backfilling predictions: %s", e)
            return

        if backfilled > 0:
            with open(PREDICTION_LOG_PATH, "w") as f:
                f.write("\n".join(updated_lines) + "\n")
            logger.info("Backfilled %d prediction outcomes", backfilled)
        self._load_pending_prediction_symbols()

    async def _log_shadow_predictions(self, max_symbols: int = 20) -> None:
        symbols = self._get_watchlist()[: max(1, int(max_symbols))]
        if not symbols:
            return
        created = 0
        for symbol in symbols:
            if symbol in self._pending_prediction_symbols:
                continue
            current_rate = await self._fetch_current_rate(symbol)
            if current_rate is None:
                continue
            confidence = min(0.99, 0.5 + abs(current_rate) * 3000.0)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "predicted_positive": bool(current_rate > 0),
                "confidence": round(float(confidence), 4),
                "predicted_rate": round(float(current_rate), 8),
                "current_rate": round(float(current_rate), 8),
                "position_size": 1.0,
                "actual_rate": None,
                "shadow": True,
            }
            with open(PREDICTION_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._pending_prediction_symbols.add(symbol)
            created += 1
        if created > 0:
            logger.info("Logged %d shadow funding predictions for learning", created)

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

    async def _fetch_current_rate(self, symbol: str) -> Optional[float]:
        """Fetch current implied funding rate (premium index)."""
        try:
            from binance.um_futures import UMFutures
            client = UMFutures(base_url="https://fapi.binance.com")
            result = client.premium_index(symbol=symbol)
            if isinstance(result, dict):
                return float(result.get("lastFundingRate", 0.0))
        except Exception as e:
            logger.debug("Failed to fetch current rate for %s: %s", symbol, e)
        return None

    async def _try_retrain(self) -> None:
        """Attempt to retrain the model on updated data."""
        now_str = datetime.now(timezone.utc).isoformat()
        symbols = self._get_watchlist()
        if not symbols:
            logger.debug("No watchlist symbols for retraining")
            self._record_retrain(
                {
                    "time": now_str,
                    "result": "skipped",
                    "reason": "empty_watchlist",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "data_rows": 0,
                    "new_rates_fetched": 0,
                }
            )
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
            self._record_retrain(
                {
                    "time": now_str,
                    "result": "rejected",
                    "reason": f"feature_build_failed: {e}",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "data_rows": 0,
                    "new_rates_fetched": total_new,
                }
            )
            return

        reject = self._validate_training_df(df)
        if reject is not None:
            self._events.append(reject)
            logger.warning("Retrain data rejected: %s", reject.get("reason"))
            self._record_retrain(
                {
                    "time": now_str,
                    "result": "rejected",
                    "reason": str(reject.get("reason", "validation_failed")),
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "data_rows": int(len(df)),
                    "new_rates_fetched": total_new,
                }
            )
            return

        if df.empty or len(df) < config.FUNDING_RETRAIN_MIN_ROWS:
            logger.info("Insufficient data for retrain: %d rows (need %d)",
                        len(df), config.FUNDING_RETRAIN_MIN_ROWS)
            self._record_retrain(
                {
                    "time": now_str,
                    "result": "skipped",
                    "reason": "insufficient_rows",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "data_rows": int(len(df)),
                    "new_rates_fetched": total_new,
                }
            )
            return

        # Step 3: Train new model (no tuning for speed)
        try:
            predictor = FundingPredictor()
            metrics = await loop.run_in_executor(
                None, lambda: predictor.train(df, tune=False)
            )
        except Exception as e:
            logger.warning("Model training failed: %s", e)
            self._record_retrain(
                {
                    "time": now_str,
                    "result": "rejected",
                    "reason": f"train_failed: {e}",
                    "new_auc": round(self._current_auc, 4),
                    "old_auc": round(self._current_auc, 4),
                    "data_rows": int(len(df)),
                    "new_rates_fetched": total_new,
                }
            )
            return

        new_auc = metrics.get("direction_auc", 0.0)
        self._quality.add_prediction(new_auc)
        sat = self._quality.saturation_rate()
        if self._quality.prediction_is_frozen():
            self._events.append(
                {
                    "kind": "funding_prediction_frozen",
                    "model_id": "funding_online_learner",
                    "window": int(config.FUNDING_FROZEN_WINDOW),
                }
            )
        if sat >= float(config.FUNDING_SATURATION_RATE_THRESHOLD):
            self._events.append(
                {
                    "kind": "funding_prediction_saturation",
                    "model_id": "funding_online_learner",
                    "window": int(config.FUNDING_SATURATION_WINDOW),
                    "rate": round(sat, 4),
                }
            )

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

        self._record_retrain(retrain_record)

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

        gate_mult, gate_edge_bump = self._quality.gate_policy()
        quality_state = self._quality.state()
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
            "total_realized_pnl": round(self._total_realized_pnl, 4),
            "gate_multiplier": round(gate_mult, 4),
            "gate_edge_bump": round(gate_edge_bump, 6),
            "retrain_history": self._retrain_history[-10:],
            **quality_state,
        }

    def _validate_training_df(self, df) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        numeric_cols = [c for c in df.columns if c != "target_direction"]
        sample_features = {}
        for col in numeric_cols:
            try:
                series = (
                    df[col]
                    .replace([float("inf"), float("-inf")], float("nan"))
                    .dropna()
                )
                if len(series) == 0:
                    continue
                x = float(series.iloc[-1])
            except Exception:
                continue
            sample_features[str(col)] = x
        if not sample_features:
            return None
        reject = self._quality.validate_features(
            sample_features,
            symbol="*",
            context="retrain_data",
        )
        if reject is not None:
            return reject
        drift_events = self._quality.update_feature_drift(sample_features, symbol="*")
        self._events.extend(drift_events)
        if "target_direction" in df.columns:
            for val in df["target_direction"].tail(200).values.tolist():
                if val not in (0, 1):
                    reject = {
                        "kind": "funding_update_rejected",
                        "model_id": "funding_online_learner",
                        "symbol": "*",
                        "context": "retrain_data",
                        "reason": "invalid_label",
                        "label": str(val),
                    }
                    return reject
        return None

    @staticmethod
    def _rate_to_prob(rate: float) -> float:
        # Smooth bounded transform to [0.01, 0.99] for calibration metrics.
        x = max(-0.05, min(0.05, float(rate)))
        p = 1.0 / (1.0 + math.exp(-x * 400.0))
        return max(0.01, min(0.99, p))

    def _load_pending_prediction_symbols(self) -> None:
        pending = set()
        if not PREDICTION_LOG_PATH.exists():
            self._pending_prediction_symbols = pending
            return
        try:
            with open(PREDICTION_LOG_PATH, "r") as f:
                for line in f:
                    row = json.loads(line.strip())
                    sym = row.get("symbol")
                    if sym and row.get("actual_rate") is None:
                        pending.add(sym)
        except Exception:
            pass
        self._pending_prediction_symbols = pending

    def _record_retrain(self, retrain_record: Dict[str, Any]) -> None:
        self._retrain_count += 1
        self._last_retrain_ts = time.monotonic()
        self._last_retrain_time = str(retrain_record.get("time"))
        self._last_retrain_result = str(retrain_record.get("result"))
        self._retrain_history.append(dict(retrain_record))
        if len(self._retrain_history) > 50:
            self._retrain_history = self._retrain_history[-50:]


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


class SharedFundingLearnerView:
    """Read-only view over the shared funding learner artifacts."""

    def __init__(self) -> None:
        self._running = False
        self._quality = FundingLearningQuality(
            model_id="funding_online_learner",
            model_family="funding",
            state_path=QUALITY_STATE_PATH,
        )

    async def run(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(60)

    def stop(self) -> None:
        self._running = False

    def drain_events(self) -> List[Dict[str, Any]]:
        return []

    def get_state(self) -> Dict[str, Any]:
        quality_state = self._quality.state()
        gate_mult, gate_edge_bump = self._quality.gate_policy()
        current_auc = 0.0
        if META_PATH.exists():
            try:
                meta = json.loads(META_PATH.read_text(encoding="utf-8"))
                current_auc = float((meta.get("metrics") or {}).get("direction_auc", 0.0) or 0.0)
            except Exception:
                current_auc = 0.0
        return {
            "running": self._running,
            "shared_reader": True,
            "current_auc": round(current_auc, 4),
            "retrain_count": None,
            "last_retrain_time": None,
            "last_retrain_result": "shared_reader",
            "next_retrain_hours": None,
            "retrain_interval_hours": config.FUNDING_RETRAIN_INTERVAL_HOURS,
            "min_auc_threshold": config.FUNDING_RETRAIN_MIN_AUC,
            "prediction_accuracy": 0.0,
            "prediction_total": 0,
            "total_realized_pnl": 0.0,
            "gate_multiplier": round(gate_mult, 4),
            "gate_edge_bump": round(gate_edge_bump, 6),
            "retrain_history": [],
            **quality_state,
        }
