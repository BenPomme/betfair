"""
Risk manager for funding rate arbitrage.
Approves or rejects opportunities based on exposure limits, leverage,
liquidation distance, and position constraints.

OPUS-reviewed: Financially critical — guards against excessive risk.
"""
import logging
from decimal import Decimal
from typing import List, Optional, Tuple

import config
from funding.core.schemas import (
    ContrarianSignal,
    DirectionalPosition,
    DirectionalPositionStatus,
    FundingOpportunity,
    HedgePosition,
    HedgeStatus,
)

logger = logging.getLogger(__name__)

# Circuit breaker state
_consecutive_failures: int = 0
CIRCUIT_BREAKER_THRESHOLD: int = 3
trading_halted: bool = False


def reset_circuit_breaker() -> None:
    """Manually reset the circuit breaker."""
    global _consecutive_failures, trading_halted
    _consecutive_failures = 0
    trading_halted = False
    logger.info("Funding circuit breaker reset")


def record_failure() -> bool:
    """Record an execution failure. Returns True if circuit breaker trips."""
    global _consecutive_failures, trading_halted
    _consecutive_failures += 1
    logger.warning(
        "Funding execution failure %d/%d",
        _consecutive_failures, CIRCUIT_BREAKER_THRESHOLD,
    )
    if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        trading_halted = True
        logger.error("Funding circuit breaker TRIPPED — trading halted")
        return True
    return False


def record_success() -> None:
    """Record a successful execution, resetting failure counter."""
    global _consecutive_failures
    _consecutive_failures = 0


def approve(
    opportunity: FundingOpportunity,
    open_positions: List[HedgePosition],
    liquidation_price: Optional[Decimal] = None,
    max_total_exposure: Optional[Decimal] = None,
    max_open_hedges: Optional[int] = None,
    max_position: Optional[Decimal] = None,
    max_leverage: Optional[int] = None,
    min_liquidation_distance: Optional[Decimal] = None,
) -> Tuple[bool, str]:
    """Evaluate whether an opportunity should be executed.

    Args:
        opportunity: The detected funding opportunity.
        open_positions: Currently open hedge positions.
        liquidation_price: Calculated liquidation price for the short perp.
        max_total_exposure: Maximum total notional exposure (default: config).
        max_open_hedges: Maximum simultaneous hedge positions (default: config).
        max_position: Maximum single position size (default: config).
        max_leverage: Maximum allowed leverage (default: config).
        min_liquidation_distance: Minimum distance to liquidation (default: config).

    Returns:
        Tuple of (approved: bool, reason: str).
    """
    max_total_exposure = max_total_exposure or config.FUNDING_MAX_TOTAL_EXPOSURE_USD
    max_open_hedges = max_open_hedges or config.FUNDING_MAX_OPEN_HEDGES
    max_position = max_position or config.FUNDING_MAX_POSITION_USD
    max_leverage = max_leverage or config.FUNDING_MAX_LEVERAGE
    min_liquidation_distance = min_liquidation_distance or config.FUNDING_MIN_LIQUIDATION_DISTANCE

    # Check 0: Circuit breaker
    if trading_halted:
        return False, "trading_halted: circuit breaker tripped"

    # Check 1: Kill switch
    if config.FUNDING_KILL_SWITCH:
        return False, "kill_switch: FUNDING_KILL_SWITCH is active"

    # Check 2: Maximum open hedges
    active = [p for p in open_positions if p.status == HedgeStatus.OPEN]
    if len(active) >= max_open_hedges:
        return False, f"max_hedges: {len(active)}/{max_open_hedges} open positions"

    # Check 3: No duplicate symbol
    active_symbols = {p.symbol for p in active}
    if opportunity.symbol in active_symbols:
        return False, f"duplicate: already hedged on {opportunity.symbol}"

    # Check 4: Single position size
    if opportunity.position_size > max_position:
        return False, (
            f"position_size: ${opportunity.position_size} exceeds "
            f"max ${max_position}"
        )

    # Check 5: Total exposure
    current_exposure = sum(p.notional_value() for p in active)
    new_exposure = current_exposure + opportunity.position_size
    if new_exposure > max_total_exposure:
        return False, (
            f"total_exposure: ${new_exposure} would exceed "
            f"max ${max_total_exposure}"
        )

    # Check 6: Leverage
    if config.FUNDING_LEVERAGE > max_leverage:
        return False, (
            f"leverage: configured {config.FUNDING_LEVERAGE}x exceeds "
            f"max {max_leverage}x"
        )

    # Check 7: Liquidation distance
    if liquidation_price is not None and opportunity.entry_price_perp > Decimal("0"):
        distance = abs(
            liquidation_price - opportunity.entry_price_perp
        ) / opportunity.entry_price_perp
        if distance < min_liquidation_distance:
            return False, (
                f"liquidation: distance {distance:.2%} below "
                f"minimum {min_liquidation_distance:.2%}"
            )

    logger.debug(
        "Approved %s: size=$%s, exposure=$%s/$%s, hedges=%d/%d",
        opportunity.symbol,
        opportunity.position_size,
        new_exposure,
        max_total_exposure,
        len(active) + 1,
        max_open_hedges,
    )
    return True, "approved"


def approve_directional(
    signal: ContrarianSignal,
    open_directional: List[DirectionalPosition],
    open_hedges: List[HedgePosition],
    max_directional: Optional[int] = None,
    max_capital_pct: Optional[Decimal] = None,
    total_balance: Optional[Decimal] = None,
) -> Tuple[bool, str]:
    """Evaluate whether a contrarian directional trade should be executed.

    Enforces circuit breaker, kill switch, position count limits, duplicate
    symbol prevention, capital exposure cap, and hedge conflict checks.

    Args:
        signal: The contrarian signal requesting a directional position.
        open_directional: Currently open directional positions.
        open_hedges: Currently open hedge positions.
        max_directional: Max simultaneous directional positions
            (default: config.CONTRARIAN_MAX_POSITIONS).
        max_capital_pct: Max fraction of total_balance allocated to
            directional positions combined (default: Decimal("0.10")).
        total_balance: Total account balance in USD; required for capital
            cap check. If None the capital cap check is skipped.

    Returns:
        Tuple of (approved: bool, reason: str).
    """
    max_directional = max_directional if max_directional is not None else config.CONTRARIAN_MAX_POSITIONS
    max_capital_pct = max_capital_pct if max_capital_pct is not None else Decimal("0.10")

    # Check 1: Circuit breaker
    if trading_halted:
        return False, "trading_halted: circuit breaker tripped"

    # Check 2: Kill switch
    if config.FUNDING_KILL_SWITCH:
        return False, "kill_switch: FUNDING_KILL_SWITCH is active"

    # Filter to active (OPEN) directional positions only
    active_directional = [
        p for p in open_directional
        if p.status == DirectionalPositionStatus.OPEN
    ]

    # Check 3: Max simultaneous directional positions
    if len(active_directional) >= max_directional:
        return False, (
            f"max_directional: {len(active_directional)}/{max_directional} "
            "directional positions already open"
        )

    # Check 4: No duplicate symbol in active directional positions
    active_directional_symbols = {p.symbol for p in active_directional}
    if signal.symbol in active_directional_symbols:
        return False, f"duplicate: already have a directional position on {signal.symbol}"

    # Check 5: Capital exposure cap
    if total_balance is not None and total_balance > Decimal("0"):
        current_directional_exposure = sum(
            p.notional_value() for p in active_directional
        )
        exposure_ratio = current_directional_exposure / total_balance
        if exposure_ratio >= max_capital_pct:
            return False, (
                f"capital_cap: directional exposure {exposure_ratio:.2%} "
                f"already at or above cap {max_capital_pct:.2%} of balance"
            )

    # Check 6: No conflicting open hedge on the same symbol
    active_hedges = [p for p in open_hedges if p.status == HedgeStatus.OPEN]
    active_hedge_symbols = {p.symbol for p in active_hedges}
    if signal.symbol in active_hedge_symbols:
        return False, (
            f"hedge_conflict: {signal.symbol} is already open as a hedge position; "
            "cannot simultaneously hold a directional position on the same symbol"
        )

    logger.debug(
        "Approved directional %s %s: active=%d/%d",
        signal.symbol,
        signal.direction.value,
        len(active_directional) + 1,
        max_directional,
    )
    return True, "approved"
