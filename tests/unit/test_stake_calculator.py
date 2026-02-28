"""Unit tests for core.stake_calculator. Equal-profit stakes, 2 dp rounding."""
import pytest
from decimal import Decimal

from core.stake_calculator import compute_stakes
from core.commission import MBR


def test_compute_stakes_returns_stakes_and_net_profits():
    prices = [Decimal("2.50"), Decimal("3.50"), Decimal("3.20")]
    total_stake = Decimal("100.00")
    result = compute_stakes(prices, total_stake, MBR, Decimal("0"))
    assert result is not None
    stakes, net_profits, min_net = result
    assert len(stakes) == 3
    assert sum(stakes) == total_stake
    assert all(s > 0 for s in stakes)
    assert all(n > 0 for n in net_profits)
    assert min_net == min(net_profits)


def test_compute_stakes_equal_profit_across_outcomes():
    # With equal-profit formula, net profit (after commission) should be equal
    # for each outcome when we use the same stakes
    prices = [Decimal("2.10"), Decimal("2.10")]
    total_stake = Decimal("100.00")
    result = compute_stakes(prices, total_stake, MBR, Decimal("0"))
    assert result is not None
    _, net_profits, _ = result
    assert len(net_profits) == 2
    assert net_profits[0] == net_profits[1]


def test_compute_stakes_rounding_two_decimal_places():
    # Use prices with overround < 1 so arb is profitable
    prices = [Decimal("2.50"), Decimal("3.50"), Decimal("3.20")]
    total_stake = Decimal("99.99")
    result = compute_stakes(prices, total_stake, MBR, Decimal("0"))
    assert result is not None
    stakes, _, _ = result
    for s in stakes:
        assert s.as_tuple().exponent >= -2  # at most 2 decimal places
    assert sum(stakes) == total_stake


def test_compute_stakes_not_profitable_returns_none():
    # Overround >= 1 so no arb
    prices = [Decimal("1.90"), Decimal("2.00")]
    total_stake = Decimal("100.00")
    result = compute_stakes(prices, total_stake, MBR, Decimal("0"))
    assert result is None
