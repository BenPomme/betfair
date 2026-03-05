"""
Exit strategy for funding rate arbitrage.
Phase 1: rule-based exits.
Phase 2: ML-assisted exits (predict negative rate with high confidence).
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import config
from funding.core.schemas import FundingSnapshot, HedgePosition, HedgeStatus
from funding.core import hedge_calculator

logger = logging.getLogger(__name__)


def evaluate_exits(
    open_positions: List[HedgePosition],
    snapshots: Dict[str, FundingSnapshot],
    max_hold_hours: Optional[int] = None,
    min_liquidation_distance: Optional[Decimal] = None,
    ml_predictor=None,
    ml_exit_confidence: float = 0.65,
    rate_histories: Optional[Dict[str, list]] = None,
) -> List[str]:
    """Evaluate which open positions should be closed.

    Exit signals (checked in order):
    1. Current funding rate is negative (funding flip)
    2. Position held > max_hold_hours
    3. Liquidation distance < minimum
    4. ML predicts negative rate with high confidence (Phase 2)

    Args:
        open_positions: Currently open hedge positions.
        snapshots: Current price/rate data per symbol.
        max_hold_hours: Max hold duration (default: config).
        min_liquidation_distance: Min liquidation distance (default: config).
        ml_predictor: Optional FundingPredictor for ML-assisted exits.
        ml_exit_confidence: Confidence threshold for ML exit signal.
        rate_histories: Optional dict of symbol -> rate history for ML features.

    Returns:
        List of symbols to close.
    """
    max_hold = max_hold_hours or config.FUNDING_MAX_HOLD_HOURS
    min_liq_dist = min_liquidation_distance or config.FUNDING_MIN_LIQUIDATION_DISTANCE

    exits: List[str] = []
    now = datetime.now(timezone.utc)

    for pos in open_positions:
        if pos.status != HedgeStatus.OPEN:
            continue

        symbol = pos.symbol
        reason = None

        # Check 1: Funding rate flip (negative = we pay)
        snapshot = snapshots.get(symbol)
        if snapshot and snapshot.funding_rate < Decimal("0"):
            reason = f"rate_flip: funding rate {snapshot.funding_rate} is negative"

        # Check 2: Max hold duration
        if reason is None and pos.entry_time:
            entry = pos.entry_time
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            hours_held = (now - entry).total_seconds() / 3600
            if hours_held > max_hold:
                reason = f"max_hold: held {hours_held:.1f}h > {max_hold}h limit"

        # Check 3: Liquidation distance
        if reason is None and snapshot:
            liq_price = hedge_calculator.calculate_liquidation_price(
                pos.entry_price_perp, pos.leverage
            )
            is_safe, distance = hedge_calculator.check_liquidation_distance(
                snapshot.mark_price, liq_price, min_liq_dist
            )
            if not is_safe:
                reason = (
                    f"liquidation: distance {distance:.2%} below "
                    f"minimum {min_liq_dist:.2%}"
                )

        # Check 4: ML predicts negative rate with high confidence
        if reason is None and ml_predictor and rate_histories:
            history = rate_histories.get(symbol, [])
            if len(history) >= 4:
                try:
                    from funding.strategy.entry_strategy import _build_live_features
                    features = _build_live_features(
                        type("Opp", (), {"symbol": symbol})(),
                        history,
                    )
                    if features is not None:
                        prediction = ml_predictor.predict(features)
                        pred_positive = prediction["predicted_positive"].iloc[0]
                        confidence = prediction["confidence"].iloc[0]
                        if pred_positive == 0 and confidence >= ml_exit_confidence:
                            reason = (
                                f"ml_exit: predicts negative (conf={confidence:.2f})"
                            )
                except Exception as e:
                    logger.debug("ML exit check failed for %s: %s", symbol, e)

        if reason:
            logger.info("Exit signal for %s: %s", symbol, reason)
            exits.append(symbol)

    return exits
