"""
Hedge calculator: compute exact spot/perp quantities for delta-neutral positions.
Respects lot size filters from exchange info and calculates liquidation distance.
"""
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Dict, Optional, Tuple

import config
from funding.core.schemas import FundingOpportunity

logger = logging.getLogger(__name__)


def _round_to_step(value: Decimal, step_size: Decimal) -> Decimal:
    """Round down to nearest step size (Binance lot size filter)."""
    if step_size <= Decimal("0"):
        return value
    return (value / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size


def calculate_quantities(
    opportunity: FundingOpportunity,
    exchange_filters: Optional[Dict] = None,
) -> Tuple[Decimal, Decimal]:
    """Calculate exact spot and perp quantities for a hedge.

    Args:
        opportunity: The detected funding opportunity.
        exchange_filters: Filter dict from exchangeInfo (LOT_SIZE, etc.).

    Returns:
        Tuple of (spot_quantity, perp_quantity).
    """
    spot_price = opportunity.entry_price_spot
    if spot_price <= Decimal("0"):
        return Decimal("0"), Decimal("0")

    # Raw quantity from notional / price
    raw_qty = opportunity.position_size / spot_price

    # Apply lot size filter if available
    if exchange_filters:
        lot_size = exchange_filters.get("LOT_SIZE", {})
        step_size = Decimal(str(lot_size.get("stepSize", "0")))
        min_qty = Decimal(str(lot_size.get("minQty", "0")))
        max_qty = Decimal(str(lot_size.get("maxQty", "99999999")))

        raw_qty = _round_to_step(raw_qty, step_size)
        if raw_qty < min_qty:
            logger.warning(
                "%s: calculated qty %s below min %s",
                opportunity.symbol, raw_qty, min_qty,
            )
            return Decimal("0"), Decimal("0")
        if raw_qty > max_qty:
            raw_qty = max_qty

    # Spot and perp quantities must match for delta-neutral
    spot_qty = raw_qty
    perp_qty = raw_qty

    return spot_qty, perp_qty


def calculate_required_usdt(
    opportunity: FundingOpportunity,
    spot_qty: Decimal,
    perp_qty: Decimal,
    leverage: int = 0,
    buffer_pct: Decimal = Decimal("0.02"),
) -> Decimal:
    """Calculate total USDT needed for the hedge.

    Args:
        opportunity: The funding opportunity.
        spot_qty: Spot quantity to buy.
        perp_qty: Perp quantity to short.
        leverage: Leverage to use (0 = use config default).
        buffer_pct: Extra buffer percentage (default 2%).

    Returns:
        Total USDT required.
    """
    lev = leverage or config.FUNDING_LEVERAGE
    spot_cost = spot_qty * opportunity.entry_price_spot
    perp_margin = (perp_qty * opportunity.entry_price_perp) / Decimal(str(lev))
    subtotal = spot_cost + perp_margin
    buffer = subtotal * buffer_pct
    total = (subtotal + buffer).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return total


def calculate_liquidation_price(
    entry_price: Decimal,
    leverage: int = 0,
    margin_type: str = "ISOLATED",
    is_short: bool = True,
) -> Decimal:
    """Estimate liquidation price for a perpetual position.

    Simplified formula for ISOLATED margin:
      Short: liq_price ≈ entry_price × (1 + 1/leverage - maintenance_margin_rate)
      Long:  liq_price ≈ entry_price × (1 - 1/leverage + maintenance_margin_rate)

    Maintenance margin rate is ~0.4% for most symbols at low notional.

    Args:
        entry_price: Entry price of the perp position.
        leverage: Leverage used (0 = config default).
        margin_type: ISOLATED or CROSSED.
        is_short: True for short position.

    Returns:
        Estimated liquidation price.
    """
    lev = leverage or config.FUNDING_LEVERAGE
    maintenance_rate = Decimal("0.004")  # 0.4% typical

    if is_short:
        liq = entry_price * (Decimal("1") + Decimal("1") / Decimal(str(lev)) - maintenance_rate)
    else:
        liq = entry_price * (Decimal("1") - Decimal("1") / Decimal(str(lev)) + maintenance_rate)

    return liq.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def check_liquidation_distance(
    entry_price: Decimal,
    liquidation_price: Decimal,
    min_distance: Optional[Decimal] = None,
) -> Tuple[bool, Decimal]:
    """Check if liquidation price is safely far from entry.

    Args:
        entry_price: Entry price of the perp.
        liquidation_price: Calculated liquidation price.
        min_distance: Minimum distance ratio (default: config value).

    Returns:
        Tuple of (is_safe: bool, distance_ratio: Decimal).
    """
    min_distance = min_distance or config.FUNDING_MIN_LIQUIDATION_DISTANCE
    if entry_price <= Decimal("0"):
        return False, Decimal("0")

    distance = abs(liquidation_price - entry_price) / entry_price
    is_safe = distance >= min_distance
    return is_safe, distance
