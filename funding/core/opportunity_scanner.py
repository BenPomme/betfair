"""
Opportunity scanner for funding rate arbitrage.
Scans all cached FundingSnapshots and identifies profitable hedging opportunities.

OPUS-reviewed: This is core detection logic — financially critical.

Detection criteria:
  1. Funding rate > FUNDING_MIN_RATE_PER_8H
  2. Annualized yield > FUNDING_MIN_ANNUALIZED_YIELD
  3. 24h quote volume > FUNDING_MIN_24H_VOLUME_USD
  4. Position size capped at FUNDING_MAX_POSITION_USD
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

import config
from funding.core import fee_calculator
from funding.core.schemas import FundingOpportunity, FundingSnapshot

logger = logging.getLogger(__name__)


def proportional_position_size(
    remaining_exposure: Decimal,
    remaining_slots: int,
    max_position: Optional[Decimal] = None,
) -> Decimal:
    """Equal-weight the remaining hedge budget across remaining slots.

    This keeps new entries proportional to the remaining deployable capital
    instead of blindly using a fixed notional for every symbol.
    """
    if remaining_exposure <= Decimal("0") or remaining_slots <= 0:
        return Decimal("0")
    max_position = max_position or config.FUNDING_MAX_POSITION_USD
    suggested = (remaining_exposure / Decimal(str(remaining_slots))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return min(max_position, suggested)


def scan_opportunities(
    snapshots: Dict[str, FundingSnapshot],
    volume_data: Dict[str, Decimal],
    watchlist: Optional[set] = None,
    min_rate: Optional[Decimal] = None,
    min_annualized: Optional[Decimal] = None,
    min_volume: Optional[Decimal] = None,
    max_position: Optional[Decimal] = None,
) -> List[FundingOpportunity]:
    """Scan all symbols for funding rate arbitrage opportunities.

    Args:
        snapshots: Current FundingSnapshot per symbol from price cache.
        volume_data: 24h quote volume in USD per symbol.
        watchlist: Optional set of symbols to restrict scanning to.
        min_rate: Minimum funding rate per 8h (default: config value).
        min_annualized: Minimum annualized yield (default: config value).
        min_volume: Minimum 24h volume USD (default: config value).
        max_position: Maximum position size USD (default: config value).

    Returns:
        List of FundingOpportunity sorted by annualized yield descending.
    """
    min_rate = min_rate or config.FUNDING_MIN_RATE_PER_8H
    min_annualized = min_annualized or config.FUNDING_MIN_ANNUALIZED_YIELD
    min_volume = min_volume or config.FUNDING_MIN_24H_VOLUME_USD
    max_position = max_position or config.FUNDING_MAX_POSITION_USD

    opportunities: List[FundingOpportunity] = []
    now = datetime.now(timezone.utc)

    for symbol, snapshot in snapshots.items():
        # Skip if not in watchlist
        if watchlist and symbol not in watchlist:
            continue

        # Check 1: Funding rate must be positive and above minimum
        if snapshot.funding_rate <= Decimal("0"):
            continue
        if snapshot.funding_rate < min_rate:
            continue

        # Check 2: Annualized yield must exceed minimum
        ann_yield = fee_calculator.annualized_yield(snapshot.funding_rate)
        if ann_yield < min_annualized:
            continue

        # Check 3: 24h volume must be sufficient
        symbol_volume = volume_data.get(symbol, Decimal("0"))
        if symbol_volume < min_volume:
            logger.debug(
                "%s: volume $%s below minimum $%s",
                symbol, symbol_volume, min_volume,
            )
            continue

        # Calculate position size: capped at max_position
        position_size = max_position

        # Calculate expected funding payment per settlement
        expected_payment = fee_calculator.funding_payment(
            position_size, snapshot.funding_rate
        )

        opportunities.append(FundingOpportunity(
            symbol=symbol,
            current_rate=snapshot.funding_rate,
            predicted_rate=snapshot.funding_rate,  # Phase 1: no ML, use current rate
            annualized_yield=ann_yield,
            entry_price_spot=snapshot.index_price,
            entry_price_perp=snapshot.mark_price,
            position_size=position_size,
            expected_funding_payment=expected_payment,
            timestamp=now,
        ))

    # Sort by annualized yield descending (best opportunities first)
    opportunities.sort(key=lambda o: o.annualized_yield, reverse=True)

    if opportunities:
        logger.info(
            "Found %d funding opportunities (best: %s at %.2f%% APY)",
            len(opportunities),
            opportunities[0].symbol,
            float(opportunities[0].annualized_yield * 100),
        )

    return opportunities
