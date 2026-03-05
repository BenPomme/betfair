"""
Entry strategy for funding rate arbitrage.
Phase 1: rule-based (last 3 rates positive + entry window).
Phase 2: ML-gated (direction prediction + confidence threshold).
"""
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from funding.core.schemas import FundingOpportunity, HedgePosition
from funding.core import risk_manager
from funding.core import hedge_calculator

logger = logging.getLogger(__name__)

# Settlement times (UTC hours)
SETTLEMENT_HOURS = [0, 8, 16]

# ML model singleton (loaded once)
_ml_predictor = None
_ml_load_attempted = False


def _get_ml_predictor():
    """Lazy-load the ML predictor model."""
    global _ml_predictor, _ml_load_attempted
    if _ml_load_attempted:
        return _ml_predictor
    _ml_load_attempted = True
    try:
        from funding.ml.funding_predictor import FundingPredictor
        model_dir = Path("data/funding_models")
        meta_path = model_dir / "funding_predictor_meta.json"
        if meta_path.exists():
            predictor = FundingPredictor(model_dir=model_dir)
            predictor.load()
            _ml_predictor = predictor
            logger.info(
                "ML predictor loaded: %d features, metrics=%s",
                len(predictor.feature_columns),
                {k: v for k, v in predictor.metrics.items()
                 if not str(k).startswith("top_features")},
            )
        else:
            logger.info("No ML model found at %s, using rule-based strategy", meta_path)
    except Exception as e:
        logger.warning("Failed to load ML predictor: %s", e)
    return _ml_predictor


def reload_ml_predictor():
    """Reload the ML predictor model (called after online retraining)."""
    global _ml_predictor, _ml_load_attempted
    _ml_load_attempted = False
    _ml_predictor = None
    return _get_ml_predictor()


def _minutes_to_next_settlement(now: Optional[datetime] = None) -> int:
    """Calculate minutes until the next funding settlement."""
    now = now or datetime.now(timezone.utc)
    current_hour = now.hour
    current_minute = now.minute

    for sh in SETTLEMENT_HOURS:
        if current_hour < sh or (current_hour == sh and current_minute == 0):
            delta = (sh - current_hour) * 60 - current_minute
            return delta

    # Next settlement is tomorrow at 00:00
    delta = (24 - current_hour) * 60 - current_minute
    return delta


def _build_live_features(
    opp: FundingOpportunity,
    rate_history: list,
) -> Optional[pd.DataFrame]:
    """Build a single-row feature vector from live data for ML prediction.

    Uses the same feature names as the training pipeline so the model
    can score it directly.
    """
    if len(rate_history) < 4:
        return None

    # rate_history is newest-first from API; reverse for chronological order
    rates = [float(h["funding_rate"]) for h in reversed(rate_history)]
    prices = [float(h.get("mark_price", 0)) for h in reversed(rate_history)]

    current_rate = rates[-1]
    now = datetime.now(timezone.utc)

    features = {
        "rate_lag_1": rates[-2] if len(rates) >= 2 else 0,
        "rate_lag_2": rates[-3] if len(rates) >= 3 else 0,
        "rate_lag_3": rates[-4] if len(rates) >= 4 else 0,
        "rate_change_8h": current_rate - rates[-2] if len(rates) >= 2 else 0,
        "rate_change_24h": current_rate - rates[-4] if len(rates) >= 4 else 0,
        "rate_mean_3": np.mean(rates[-3:]) if len(rates) >= 3 else current_rate,
        "rate_std_3": np.std(rates[-3:]) if len(rates) >= 3 else 0,
        "rate_mean_9": np.mean(rates[-9:]) if len(rates) >= 9 else np.mean(rates),
        "rate_positive_streak": sum(1 for r in reversed(rates) if r > 0),
        "price_return_8h": (prices[-1] / prices[-2] - 1) if len(prices) >= 2 and prices[-2] != 0 else 0,
        "price_return_24h": (prices[-1] / prices[-4] - 1) if len(prices) >= 4 and prices[-4] != 0 else 0,
        "price_volatility_24h": np.std([
            prices[i] / prices[i - 1] - 1
            for i in range(1, min(4, len(prices)))
            if prices[i - 1] != 0
        ]) if len(prices) >= 2 else 0,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        # Extra training columns that may not be available live.
        "kline_volume": 0.0,
        "kline_quote_volume": 0.0,
        "kline_return_8h": (prices[-1] / prices[-2] - 1) if len(prices) >= 2 and prices[-2] != 0 else 0.0,
        "kline_volatility_8h": np.std(
            [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices)) if prices[i - 1] != 0]
        ) if len(prices) >= 2 else 0.0,
        "kline_taker_buy_ratio": 0.5,
        "volume_change_8h": 0.0,
        "volume_mean_3": 0.0,
        "btc_funding_rate": 0.0,
        "btc_rate_change": 0.0,
        "btc_rate_mean_3": 0.0,
        "rate_vs_btc": current_rate,
        "rate_zscore_9": (
            (current_rate - float(np.mean(rates[-9:]))) / float(np.std(rates[-9:]))
        ) if len(rates) >= 9 and float(np.std(rates[-9:])) > 0 else 0.0,
        "rate_percentile_30": (
            sum(1 for r in rates[-30:] if r <= current_rate) / max(1, len(rates[-30:]))
        ) if len(rates) >= 2 else 0.5,
    }

    df = pd.DataFrame([features], index=[pd.Timestamp(now)])
    return df


async def evaluate_entries(
    opportunities: List[FundingOpportunity],
    open_positions: List[HedgePosition],
    futures_client=None,
    entry_window_minutes: Optional[int] = None,
    use_ml: bool = True,
    ml_min_confidence: Optional[float] = None,
    ml_min_predicted_rate: Optional[float] = None,
) -> List[Tuple[FundingOpportunity, Decimal]]:
    """Evaluate which opportunities should be entered.

    When ML model is available and use_ml=True:
      1. Risk manager must approve
      2. Time within entry window
      3. ML predicts positive direction with confidence >= ml_min_confidence
      4. ML predicted rate >= ml_min_predicted_rate

    Fallback (no ML model or use_ml=False):
      1. Risk manager must approve
      2. Time within entry window
      3. Last 3 funding rates all positive

    Returns:
        List of (opportunity, position_size) to execute.
    """
    entry_window = entry_window_minutes or config.FUNDING_ENTRY_WINDOW_MINUTES
    ml_min_confidence = ml_min_confidence if ml_min_confidence is not None else config.FUNDING_ML_MIN_CONFIDENCE
    ml_min_predicted_rate = ml_min_predicted_rate if ml_min_predicted_rate is not None else config.FUNDING_ML_MIN_PREDICTED_RATE
    if str(getattr(config, "FUNDING_MODE", "paper")).lower() == "paper":
        # In paper mode we widen exploration to generate real settled labels faster.
        entry_window = max(int(entry_window), 120)
        ml_min_confidence = min(float(ml_min_confidence), 0.60)
        ml_min_predicted_rate = min(float(ml_min_predicted_rate), 0.00005)
    entries: List[Tuple[FundingOpportunity, Decimal]] = []

    # Check timing
    minutes_to_settlement = _minutes_to_next_settlement()
    if minutes_to_settlement > entry_window:
        logger.debug(
            "Outside entry window: %d min to settlement (window=%d min)",
            minutes_to_settlement, entry_window,
        )
        return entries

    # Try to load ML model
    predictor = _get_ml_predictor() if use_ml else None

    for opp in opportunities:
        # Check risk manager
        liq_price = hedge_calculator.calculate_liquidation_price(
            opp.entry_price_perp
        )
        approved, reason = risk_manager.approve(opp, open_positions, liq_price)
        if not approved:
            logger.debug("Risk rejected %s: %s", opp.symbol, reason)
            continue

        # Fetch rate history (needed for both ML and rule-based)
        rate_history = []
        if futures_client:
            try:
                rate_history = await futures_client.get_funding_rate_history(
                    opp.symbol, limit=10
                )
            except Exception as e:
                logger.warning("Failed to fetch rate history for %s: %s", opp.symbol, e)
                continue

        # ML-gated entry
        if predictor and rate_history:
            features = _build_live_features(opp, rate_history)
            if features is not None:
                try:
                    prediction = predictor.predict(features)
                    pred_positive = prediction["predicted_positive"].iloc[0]
                    confidence = prediction["confidence"].iloc[0]
                    predicted_rate = prediction["predicted_rate"].iloc[0]

                    if pred_positive != 1:
                        logger.debug(
                            "%s: ML predicts negative direction (conf=%.2f), skip",
                            opp.symbol, confidence,
                        )
                        continue
                    if confidence < ml_min_confidence:
                        logger.debug(
                            "%s: ML confidence %.2f < %.2f threshold, skip",
                            opp.symbol, confidence, ml_min_confidence,
                        )
                        continue
                    if predicted_rate < ml_min_predicted_rate:
                        logger.debug(
                            "%s: ML predicted rate %.6f < %.6f threshold, skip",
                            opp.symbol, predicted_rate, ml_min_predicted_rate,
                        )
                        continue

                    entries.append((opp, opp.position_size))
                    logger.info(
                        "ML entry signal: %s rate=%.4f%% predicted_rate=%.6f conf=%.2f size=$%s",
                        opp.symbol,
                        float(opp.current_rate * 100),
                        predicted_rate,
                        confidence,
                        opp.position_size,
                    )
                    # Log prediction for online learning feedback
                    try:
                        from funding.ml.online_learner import log_prediction
                        log_prediction(
                            symbol=opp.symbol,
                            predicted_positive=bool(pred_positive),
                            confidence=float(confidence),
                            predicted_rate=float(predicted_rate),
                            current_rate=float(opp.current_rate),
                            position_size=float(opp.position_size),
                        )
                    except Exception:
                        pass
                    continue
                except Exception as e:
                    logger.warning("ML prediction failed for %s: %s, falling back to rules", opp.symbol, e)

        # Rule-based fallback: last 3 rates all positive
        if rate_history and len(rate_history) >= 3:
            all_positive = all(
                h["funding_rate"] > Decimal("0") for h in rate_history[:3]
            )
            if not all_positive:
                logger.debug("%s: not all last 3 rates positive, skipping", opp.symbol)
                continue

        entries.append((opp, opp.position_size))
        logger.info(
            "Rule-based entry signal: %s at %.4f%% rate, $%s size",
            opp.symbol,
            float(opp.current_rate * 100),
            opp.position_size,
        )

    return entries
