"""
Fee calculator for Binance funding rate arbitrage.
All math uses decimal.Decimal — no floats for monetary values.

Fee structure (VIP 0):
  - Futures maker: 0.02%
  - Futures taker: 0.05%
  - Spot maker: 0.10%
  - Spot taker: 0.10%

BNB discount:
  - Futures: 10% off
  - Spot: 25% off
"""
import decimal
from decimal import Decimal, ROUND_HALF_UP

decimal.getcontext().prec = 10

# Base fee rates (VIP 0)
FUTURES_MAKER: Decimal = Decimal("0.0002")   # 0.02%
FUTURES_TAKER: Decimal = Decimal("0.0005")   # 0.05%
SPOT_MAKER: Decimal = Decimal("0.001")       # 0.10%
SPOT_TAKER: Decimal = Decimal("0.001")       # 0.10%

# BNB discount factors (multiply fee by (1 - discount))
BNB_FUTURES_DISCOUNT: Decimal = Decimal("0.10")  # 10% off
BNB_SPOT_DISCOUNT: Decimal = Decimal("0.25")     # 25% off


def _apply_bnb_discount(rate: Decimal, is_futures: bool, bnb_discount: bool) -> Decimal:
    """Apply BNB fee discount if enabled."""
    if not bnb_discount:
        return rate
    discount = BNB_FUTURES_DISCOUNT if is_futures else BNB_SPOT_DISCOUNT
    return (rate * (Decimal("1") - discount)).quantize(Decimal("0.0000001"), rounding=ROUND_HALF_UP)


def spot_fee(notional: Decimal, maker: bool = False, bnb_discount: bool = False) -> Decimal:
    """Calculate fee for a single spot trade.

    Args:
        notional: Trade notional value in USD.
        maker: True for limit orders (maker rate), False for market orders (taker rate).
        bnb_discount: True if paying fees in BNB.

    Returns:
        Fee amount in USD (Decimal, 2 decimal places).
    """
    rate = SPOT_MAKER if maker else SPOT_TAKER
    rate = _apply_bnb_discount(rate, is_futures=False, bnb_discount=bnb_discount)
    return (notional * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def futures_fee(notional: Decimal, maker: bool = False, bnb_discount: bool = False) -> Decimal:
    """Calculate fee for a single futures trade.

    Args:
        notional: Trade notional value in USD.
        maker: True for limit orders (maker rate), False for market orders (taker rate).
        bnb_discount: True if paying fees in BNB.

    Returns:
        Fee amount in USD (Decimal, 2 decimal places).
    """
    rate = FUTURES_MAKER if maker else FUTURES_TAKER
    rate = _apply_bnb_discount(rate, is_futures=True, bnb_discount=bnb_discount)
    return (notional * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def trading_fees_round_trip(
    notional: Decimal,
    maker: bool = False,
    bnb_discount: bool = False,
) -> Decimal:
    """Total trading fees for entering AND exiting both legs (spot + perp).

    A full round trip = 4 trades:
      1. Spot buy (entry)
      2. Perp short (entry)
      3. Perp close / buy (exit)
      4. Spot sell (exit)

    Args:
        notional: Position notional value in USD.
        maker: True for limit orders on all legs, False for market orders.
        bnb_discount: True if paying fees in BNB.

    Returns:
        Total fees in USD (Decimal, 2 decimal places).
    """
    spot_entry = spot_fee(notional, maker=maker, bnb_discount=bnb_discount)
    spot_exit = spot_fee(notional, maker=maker, bnb_discount=bnb_discount)
    perp_entry = futures_fee(notional, maker=maker, bnb_discount=bnb_discount)
    perp_exit = futures_fee(notional, maker=maker, bnb_discount=bnb_discount)
    return spot_entry + spot_exit + perp_entry + perp_exit


def funding_payment(notional: Decimal, rate: Decimal) -> Decimal:
    """Calculate a single funding settlement payment.

    Positive rate + short position = you RECEIVE funding.
    Negative rate + short position = you PAY funding.

    Args:
        notional: Position notional value at mark price.
        rate: Funding rate for the settlement period (e.g. 0.0001 = 0.01%).

    Returns:
        Payment amount in USD. Positive = received, negative = paid.
    """
    return (notional * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def breakeven_periods(
    notional: Decimal,
    rate_per_8h: Decimal,
    maker: bool = False,
    bnb_discount: bool = False,
) -> int:
    """Number of 8h funding periods needed to recoup round-trip trading fees.

    Args:
        notional: Position notional value in USD.
        rate_per_8h: Expected funding rate per 8h settlement.
        maker: True if using limit orders.
        bnb_discount: True if paying fees in BNB.

    Returns:
        Number of 8h periods (int, rounded up). Returns -1 if rate is zero or negative.
    """
    if rate_per_8h <= Decimal("0"):
        return -1
    total_fees = trading_fees_round_trip(notional, maker=maker, bnb_discount=bnb_discount)
    payment_per_period = funding_payment(notional, rate_per_8h)
    if payment_per_period <= Decimal("0"):
        return -1
    # Ceiling division
    periods = total_fees / payment_per_period
    return int(periods.to_integral_value(rounding=decimal.ROUND_CEILING))


def annualized_yield(rate_per_8h: Decimal) -> Decimal:
    """Convert an 8h funding rate to annualized yield.

    Formula: rate_per_8h × 3 (settlements/day) × 365 (days/year)

    Args:
        rate_per_8h: Funding rate per 8h settlement.

    Returns:
        Annualized yield as Decimal (e.g. 0.1095 = 10.95%).
    """
    return (rate_per_8h * Decimal("3") * Decimal("365")).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )


def net_yield_after_fees(
    notional: Decimal,
    rate_per_8h: Decimal,
    hold_periods: int,
    maker: bool = False,
    bnb_discount: bool = False,
) -> Decimal:
    """Net yield after holding for a given number of funding periods.

    Args:
        notional: Position notional value in USD.
        rate_per_8h: Expected average funding rate per settlement.
        hold_periods: Number of 8h settlement periods to hold.
        maker: True if using limit orders.
        bnb_discount: True if paying fees in BNB.

    Returns:
        Net profit/loss in USD.
    """
    total_funding = funding_payment(notional, rate_per_8h) * hold_periods
    total_fees = trading_fees_round_trip(notional, maker=maker, bnb_discount=bnb_discount)
    return total_funding - total_fees
