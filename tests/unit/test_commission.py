"""Unit tests for core.commission. All monetary values via Decimal."""
import pytest
from decimal import Decimal

from core.commission import (
    effective_rate,
    commission,
    evaluate_back_back_arb,
    evaluate_lay_lay_arb,
    MBR,
)


def test_effective_rate_zero_discount():
    mbr = Decimal("0.05")
    discount = Decimal("0")
    assert effective_rate(mbr, discount) == Decimal("0.05")


def test_effective_rate_60_percent_discount():
    mbr = Decimal("0.05")
    discount = Decimal("0.60")
    assert effective_rate(mbr, discount) == Decimal("0.02")


def test_commission_zero_discount():
    net_winnings = Decimal("10.00")
    assert commission(net_winnings, Decimal("0.05"), Decimal("0")) == Decimal("0.50")


def test_commission_60_percent_discount():
    net_winnings = Decimal("10.00")
    assert commission(net_winnings, Decimal("0.05"), Decimal("0.60")) == Decimal("0.20")


def test_commission_rounds_half_up():
    net_winnings = Decimal("10.555")
    # 10.555 * 0.05 = 0.52775 -> 0.53
    assert commission(net_winnings, Decimal("0.05"), Decimal("0")) == Decimal("0.53")


def test_evaluate_back_back_arb_two_outcome_profitable():
    prices = [Decimal("2.10"), Decimal("2.10")]  # overround < 1
    total_stake = Decimal("100.00")
    result = evaluate_back_back_arb(prices, total_stake, MBR, Decimal("0"))
    assert result is not None
    assert result["overround"] < 1
    assert all(s > 0 for s in result["stakes"])
    assert sum(result["stakes"]) == total_stake
    assert all(n > 0 for n in result["net_profits"])
    assert result["min_net_profit"] > 0
    assert result["roi"] > 0


def test_evaluate_back_back_arb_three_outcome_market():
    # 1/2.5 + 1/3.5 + 1/3.2 ≈ 0.4 + 0.286 + 0.3125 ≈ 0.9985
    prices = [Decimal("2.50"), Decimal("3.50"), Decimal("3.20")]
    total_stake = Decimal("100.00")
    result = evaluate_back_back_arb(prices, total_stake, MBR, Decimal("0"))
    assert result is not None
    assert len(result["stakes"]) == 3
    assert len(result["net_profits"]) == 3
    assert all(n > 0 for n in result["net_profits"])


def test_evaluate_back_back_arb_not_profitable_high_overround():
    # Overround > 1, no arb
    prices = [Decimal("2.00"), Decimal("2.00")]  # 0.5 + 0.5 = 1.0, borderline
    prices_high = [Decimal("1.90"), Decimal("2.00")]  # overround > 1
    total_stake = Decimal("100.00")
    result = evaluate_back_back_arb(prices_high, total_stake, MBR, Decimal("0"))
    assert result is None or not all(n > 0 for n in result["net_profits"])


def test_evaluate_back_back_arb_overround_exactly_at_threshold():
    # Overround exactly 1.00: 1/2 + 1/2 = 1.0 -> no profit after commission
    prices = [Decimal("2.00"), Decimal("2.00")]
    total_stake = Decimal("100.00")
    result = evaluate_back_back_arb(prices, total_stake, MBR, Decimal("0"))
    # Either None or net_profits not all positive (commission eats the edge)
    if result is not None:
        assert not all(n > 0 for n in result["net_profits"])


def test_evaluate_back_back_arb_empty_prices_returns_none():
    assert evaluate_back_back_arb([], Decimal("100"), MBR, Decimal("0")) is None


def test_evaluate_back_back_arb_zero_stake_returns_none():
    assert evaluate_back_back_arb(
        [Decimal("2.0"), Decimal("2.0")], Decimal("0"), MBR, Decimal("0")
    ) is None


# ============================================================================
# Tests for evaluate_lay_lay_arb
# ============================================================================


def test_lay_lay_arb_2_outcome_unprofitable():
    """
    Tennis: lay_prices=[1.80, 2.30], total_liability=100, mbr=0.05, discount=0.0.
    lay_overround = 1/1.80 + 1/2.30 ≈ 0.556 + 0.435 ≈ 0.991 < 1.0
    Should return None (not profitable).
    """
    lay_prices = [Decimal("1.80"), Decimal("2.30")]
    total_liability = Decimal("100.00")
    result = evaluate_lay_lay_arb(lay_prices, total_liability, MBR, Decimal("0"))
    assert result is None


def test_lay_lay_arb_3_outcome_profitable():
    """
    Football: lay_prices=[2.10, 3.20, 3.80], total_liability=100, mbr=0.05, discount=0.0.
    lay_overround = 1/2.10 + 1/3.20 + 1/3.80 ≈ 0.476 + 0.313 + 0.263 ≈ 1.052 > 1.0
    Should return a profitable dict with all net_profits > 0.
    """
    lay_prices = [Decimal("2.10"), Decimal("3.20"), Decimal("3.80")]
    total_liability = Decimal("100.00")
    result = evaluate_lay_lay_arb(lay_prices, total_liability, MBR, Decimal("0"))
    assert result is not None
    assert result["lay_overround"] > Decimal("1")
    assert len(result["stakes"]) == 3
    assert len(result["liabilities"]) == 3
    assert len(result["net_profits"]) == 3
    assert all(n > 0 for n in result["net_profits"])
    assert result["min_net_profit"] > 0
    assert result["roi"] > 0
    # Verify stakes, liabilities, and collected amounts make sense
    assert result["total_collected"] == sum(result["stakes"])
    assert all(l > 0 for l in result["liabilities"])


def test_lay_lay_arb_barely_unprofitable():
    """
    lay_prices where lay_overround is barely below 1.0 (impossible to profit).
    Try [3.00, 3.00, 3.00] → lay_overround = 1.0, should return None.
    Then try [2.999, 3.000, 3.001] → lay_overround slightly above 1.0,
    but with 5% commission, might not be profitable.
    """
    # Exactly at overround 1.0
    lay_prices_even = [Decimal("3.00"), Decimal("3.00"), Decimal("3.00")]
    result_even = evaluate_lay_lay_arb(
        lay_prices_even, Decimal("100.00"), MBR, Decimal("0")
    )
    # At exactly 1.0, should be None
    assert result_even is None


def test_lay_lay_arb_with_discount():
    """
    Same as test_lay_lay_arb_3_outcome_profitable but with discount=0.60.
    Higher net profit due to lower commission.
    """
    lay_prices = [Decimal("2.10"), Decimal("3.20"), Decimal("3.80")]
    total_liability = Decimal("100.00")
    result_no_discount = evaluate_lay_lay_arb(
        lay_prices, total_liability, MBR, Decimal("0")
    )
    result_with_discount = evaluate_lay_lay_arb(
        lay_prices, total_liability, MBR, Decimal("0.60")
    )

    assert result_no_discount is not None
    assert result_with_discount is not None

    # With discount, net profits should be higher (lower commission)
    for np_no_disc, np_with_disc in zip(
        result_no_discount["net_profits"], result_with_discount["net_profits"]
    ):
        assert np_with_disc > np_no_disc

    # Min net profit and ROI should also be better with discount
    assert result_with_discount["min_net_profit"] > result_no_discount["min_net_profit"]
    assert result_with_discount["roi"] > result_no_discount["roi"]


def test_lay_lay_arb_price_near_one():
    """
    One lay price very close to 1.0 like [1.02, 5.0, 8.0].
    The liability for the 1.02 outcome is tiny (0.02 * stake).
    Verify it handles edge case correctly.
    lay_overround ≈ 1/1.02 + 1/5.0 + 1/8.0 ≈ 0.9804 + 0.2 + 0.125 ≈ 1.3054 > 1.0
    Should be profitable.
    """
    lay_prices = [Decimal("1.02"), Decimal("5.00"), Decimal("8.00")]
    total_liability = Decimal("100.00")
    result = evaluate_lay_lay_arb(lay_prices, total_liability, MBR, Decimal("0"))
    assert result is not None
    assert result["lay_overround"] > Decimal("1")
    assert all(n > 0 for n in result["net_profits"])
    # Liability for 1.02 outcome should be very small (0.02 * stake)
    assert result["liabilities"][0] > 0
    # But stake itself should be reasonable
    assert result["stakes"][0] > 0


def test_lay_lay_arb_empty_prices_returns_none():
    """Empty lay_prices should return None."""
    assert (
        evaluate_lay_lay_arb([], Decimal("100"), MBR, Decimal("0"))
        is None
    )


def test_lay_lay_arb_zero_liability_returns_none():
    """Zero total_liability should return None."""
    assert (
        evaluate_lay_lay_arb(
            [Decimal("2.0"), Decimal("2.0")], Decimal("0"), MBR, Decimal("0")
        )
        is None
    )


def test_lay_lay_arb_overround_exactly_one():
    """
    lay_overround exactly 1.0 (e.g., [2.0, 2.0]).
    lay_overround = 1/2.0 + 1/2.0 = 1.0
    Should return None since lay_overround <= 1.0 check happens early.
    """
    lay_prices = [Decimal("2.00"), Decimal("2.00")]
    total_liability = Decimal("100.00")
    result = evaluate_lay_lay_arb(lay_prices, total_liability, MBR, Decimal("0"))
    assert result is None


def test_lay_lay_arb_two_outcome_profitable():
    """
    Two-outcome market with profitable lay overround.
    lay_prices=[1.50, 2.00]
    lay_overround = 1/1.50 + 1/2.00 ≈ 0.667 + 0.5 ≈ 1.167 > 1.0
    Should be profitable.
    """
    lay_prices = [Decimal("1.50"), Decimal("2.00")]
    total_liability = Decimal("100.00")
    result = evaluate_lay_lay_arb(lay_prices, total_liability, MBR, Decimal("0"))
    assert result is not None
    assert result["lay_overround"] > Decimal("1")
    assert len(result["stakes"]) == 2
    assert len(result["liabilities"]) == 2
    assert all(n > 0 for n in result["net_profits"])
    assert result["min_net_profit"] > 0
    assert result["roi"] > 0


def test_back_back_arb_basic():
    """
    3-outcome with prices [2.5, 3.5, 4.0], total_stake=100, mbr=0.05, discount=0.0.
    overround = 1/2.5 + 1/3.5 + 1/4.0 ≈ 0.4 + 0.286 + 0.25 ≈ 0.936 < 1.0 (profitable)
    Verify overround < 1.0, stakes sum close to 100, all net_profits > 0.
    """
    prices = [Decimal("2.50"), Decimal("3.50"), Decimal("4.00")]
    total_stake = Decimal("100.00")
    result = evaluate_back_back_arb(prices, total_stake, MBR, Decimal("0"))
    assert result is not None
    assert result["overround"] < Decimal("1")
    # Stakes should sum to (or very close to) total_stake due to rounding
    assert abs(sum(result["stakes"]) - total_stake) < Decimal("0.10")
    assert all(n > 0 for n in result["net_profits"])
    assert result["min_net_profit"] > 0
    assert result["roi"] > 0


def test_back_back_arb_unprofitable():
    """
    3-outcome with prices [1.5, 3.0, 5.0].
    overround = 1/1.5 + 1/3.0 + 1/5.0 ≈ 0.667 + 0.333 + 0.2 ≈ 1.2 > 1.0 (not profitable)
    Should return None.
    """
    prices = [Decimal("1.50"), Decimal("3.00"), Decimal("5.00")]
    total_stake = Decimal("100.00")
    result = evaluate_back_back_arb(prices, total_stake, MBR, Decimal("0"))
    assert result is None
