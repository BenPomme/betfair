"""Unit tests for funding fee calculator. All monetary values use Decimal."""
from decimal import Decimal

import pytest

from funding.core.fee_calculator import (
    spot_fee,
    futures_fee,
    trading_fees_round_trip,
    funding_payment,
    breakeven_periods,
    annualized_yield,
    net_yield_after_fees,
    FUTURES_MAKER,
    FUTURES_TAKER,
    SPOT_MAKER,
    SPOT_TAKER,
)


class TestSpotFee:
    def test_taker_fee(self):
        # $1000 notional × 0.10% = $1.00
        assert spot_fee(Decimal("1000"), maker=False) == Decimal("1.00")

    def test_maker_fee(self):
        # $1000 notional × 0.10% = $1.00 (same rate for spot)
        assert spot_fee(Decimal("1000"), maker=True) == Decimal("1.00")

    def test_bnb_discount(self):
        # $1000 × 0.10% × (1 - 0.25) = $1000 × 0.075% = $0.75
        assert spot_fee(Decimal("1000"), maker=False, bnb_discount=True) == Decimal("0.75")

    def test_small_notional(self):
        # $10 × 0.10% = $0.01
        assert spot_fee(Decimal("10"), maker=False) == Decimal("0.01")

    def test_zero_notional(self):
        assert spot_fee(Decimal("0"), maker=False) == Decimal("0.00")


class TestFuturesFee:
    def test_taker_fee(self):
        # $1000 × 0.05% = $0.50
        assert futures_fee(Decimal("1000"), maker=False) == Decimal("0.50")

    def test_maker_fee(self):
        # $1000 × 0.02% = $0.20
        assert futures_fee(Decimal("1000"), maker=True) == Decimal("0.20")

    def test_bnb_discount_taker(self):
        # $1000 × 0.05% × (1 - 0.10) = $1000 × 0.045% = $0.45
        assert futures_fee(Decimal("1000"), maker=False, bnb_discount=True) == Decimal("0.45")

    def test_bnb_discount_maker(self):
        # $1000 × 0.02% × (1 - 0.10) = $1000 × 0.018% = $0.18
        assert futures_fee(Decimal("1000"), maker=True, bnb_discount=True) == Decimal("0.18")


class TestTradingFeesRoundTrip:
    def test_taker_round_trip(self):
        # $1000 notional, 4 legs:
        # 2 × spot taker ($1.00 each) + 2 × futures taker ($0.50 each) = $3.00
        result = trading_fees_round_trip(Decimal("1000"), maker=False)
        assert result == Decimal("3.00")

    def test_maker_round_trip(self):
        # 2 × spot maker ($1.00 each) + 2 × futures maker ($0.20 each) = $2.40
        result = trading_fees_round_trip(Decimal("1000"), maker=True)
        assert result == Decimal("2.40")

    def test_bnb_discount_round_trip(self):
        # 2 × spot taker with BNB ($0.75 each) + 2 × futures taker with BNB ($0.45 each) = $2.40
        result = trading_fees_round_trip(Decimal("1000"), maker=False, bnb_discount=True)
        assert result == Decimal("2.40")

    def test_large_position(self):
        # $10000 × 0.30% total = $30.00
        result = trading_fees_round_trip(Decimal("10000"), maker=False)
        assert result == Decimal("30.00")


class TestFundingPayment:
    def test_positive_rate(self):
        # $1000 × 0.01% = $0.10
        result = funding_payment(Decimal("1000"), Decimal("0.0001"))
        assert result == Decimal("0.10")

    def test_high_positive_rate(self):
        # $1000 × 0.03% = $0.30
        result = funding_payment(Decimal("1000"), Decimal("0.0003"))
        assert result == Decimal("0.30")

    def test_negative_rate(self):
        # $1000 × -0.01% = -$0.10 (short pays when rate is negative)
        result = funding_payment(Decimal("1000"), Decimal("-0.0001"))
        assert result == Decimal("-0.10")

    def test_zero_rate(self):
        result = funding_payment(Decimal("1000"), Decimal("0"))
        assert result == Decimal("0.00")

    def test_large_notional(self):
        # $50000 × 0.05% = $25.00
        result = funding_payment(Decimal("50000"), Decimal("0.0005"))
        assert result == Decimal("25.00")


class TestBreakevenPeriods:
    def test_standard_breakeven(self):
        # $1000, taker fees = $3.00 round trip
        # Funding per period at 0.01% = $0.10
        # Breakeven = ceil(3.00 / 0.10) = 30 periods
        result = breakeven_periods(Decimal("1000"), Decimal("0.0001"))
        assert result == 30

    def test_high_rate_breakeven(self):
        # $1000, taker fees = $3.00
        # Funding per period at 0.03% = $0.30
        # Breakeven = ceil(3.00 / 0.30) = 10 periods
        result = breakeven_periods(Decimal("1000"), Decimal("0.0003"))
        assert result == 10

    def test_bnb_discount_breakeven(self):
        # $1000, BNB taker fees = $2.40
        # Funding per period at 0.03% = $0.30
        # Breakeven = ceil(2.40 / 0.30) = 8 periods
        result = breakeven_periods(Decimal("1000"), Decimal("0.0003"), bnb_discount=True)
        assert result == 8

    def test_zero_rate(self):
        result = breakeven_periods(Decimal("1000"), Decimal("0"))
        assert result == -1

    def test_negative_rate(self):
        result = breakeven_periods(Decimal("1000"), Decimal("-0.0001"))
        assert result == -1

    def test_maker_breakeven(self):
        # $1000, maker fees = $2.40
        # Funding per period at 0.01% = $0.10
        # Breakeven = ceil(2.40 / 0.10) = 24 periods
        result = breakeven_periods(Decimal("1000"), Decimal("0.0001"), maker=True)
        assert result == 24


class TestAnnualizedYield:
    def test_standard_rate(self):
        # 0.01% per 8h = 0.0001 × 3 × 365 = 0.1095 (10.95%)
        result = annualized_yield(Decimal("0.0001"))
        assert result == Decimal("0.1095")

    def test_high_rate(self):
        # 0.03% per 8h = 0.0003 × 3 × 365 = 0.3285 (32.85%)
        result = annualized_yield(Decimal("0.0003"))
        assert result == Decimal("0.3285")

    def test_very_high_rate(self):
        # 0.10% per 8h = 0.001 × 3 × 365 = 1.095 (109.5%)
        result = annualized_yield(Decimal("0.001"))
        assert result == Decimal("1.0950")

    def test_zero_rate(self):
        result = annualized_yield(Decimal("0"))
        assert result == Decimal("0.0000")

    def test_negative_rate(self):
        result = annualized_yield(Decimal("-0.0001"))
        assert result == Decimal("-0.1095")

    def test_minimum_actionable_rate(self):
        # 0.02% per 8h (FUNDING_MIN_RATE_PER_8H default)
        # 0.0002 × 3 × 365 = 0.2190 (21.90%)
        result = annualized_yield(Decimal("0.0002"))
        assert result == Decimal("0.2190")


class TestNetYieldAfterFees:
    def test_profitable_hold(self):
        # $1000, 0.03% rate, 15 periods
        # Funding: 15 × $0.30 = $4.50
        # Fees: $3.00
        # Net: $1.50
        result = net_yield_after_fees(Decimal("1000"), Decimal("0.0003"), 15)
        assert result == Decimal("1.50")

    def test_breakeven_hold(self):
        # $1000, 0.03% rate, 10 periods = breakeven
        # Funding: 10 × $0.30 = $3.00
        # Fees: $3.00
        # Net: $0.00
        result = net_yield_after_fees(Decimal("1000"), Decimal("0.0003"), 10)
        assert result == Decimal("0.00")

    def test_unprofitable_short_hold(self):
        # $1000, 0.01% rate, 5 periods
        # Funding: 5 × $0.10 = $0.50
        # Fees: $3.00
        # Net: -$2.50
        result = net_yield_after_fees(Decimal("1000"), Decimal("0.0001"), 5)
        assert result == Decimal("-2.50")

    def test_bnb_discount_profitable(self):
        # $1000, 0.03% rate, 10 periods, BNB discount
        # Funding: 10 × $0.30 = $3.00
        # Fees with BNB: $2.40
        # Net: $0.60
        result = net_yield_after_fees(
            Decimal("1000"), Decimal("0.0003"), 10, bnb_discount=True
        )
        assert result == Decimal("0.60")
