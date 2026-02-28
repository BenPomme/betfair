"""Unit tests for core.scanner. Synthetic prices; valid arb, missed, stale."""
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from core.types import PriceSnapshot, SelectionPrice
from core.scanner import scan_snapshot, scan_market


def _snapshot(market_id: str, prices: list, liquidity: Decimal = Decimal("1000")):
    selections = [
        SelectionPrice(str(i), f"Sel{i}", Decimal(str(p)), liquidity)
        for i, p in enumerate(prices)
    ]
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


def test_scanner_valid_arb_detected():
    # Overround < PRE_FILTER_THRESHOLD (0.97), profitable after commission
    # 1/3.2 + 1/3.2 + 1/3.2 = 0.9375
    prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
    snap = _snapshot("1.1", prices)
    opp = scan_snapshot(snap, event_name="Test Event")
    assert opp is not None
    assert opp.market_id == "1.1"
    assert opp.overround_raw < 1
    assert opp.net_profit_eur > 0
    assert len(opp.selections) == 3
    assert opp.total_stake_eur > 0


def test_scanner_valid_arb_missed_below_min_profit():
    # Overround slightly below 1 but min_net_profit very small; require high min
    prices = [Decimal("2.10"), Decimal("2.10")]  # small edge
    snap = _snapshot("1.2", prices)
    opp = scan_snapshot(snap, min_net_profit_eur=Decimal("100"))  # impossible
    assert opp is None


def test_scanner_false_positive_prevented_insufficient_liquidity():
    # Good prices but one selection has liquidity below MIN_LIQUIDITY
    prices = [Decimal("2.50"), Decimal("3.50"), Decimal("3.20")]
    selections = [
        SelectionPrice("0", "A", Decimal("2.50"), Decimal("1000")),
        SelectionPrice("1", "B", Decimal("3.50"), Decimal("10")),  # below default 50
        SelectionPrice("2", "C", Decimal("3.20"), Decimal("1000")),
    ]
    snap = PriceSnapshot(
        market_id="1.3",
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )
    opp = scan_snapshot(snap, min_liquidity_eur=Decimal("50"))
    assert opp is None


def test_scanner_high_overround_returns_none():
    prices = [Decimal("1.90"), Decimal("2.00")]  # overround > 1
    snap = _snapshot("1.4", prices)
    opp = scan_snapshot(snap)
    assert opp is None


def test_scan_market_returns_none_when_cache_misses():
    def get_prices(market_id):
        return None
    opp = scan_market(get_prices, "9.9")
    assert opp is None


def test_scan_market_returns_opportunity_when_cache_hit():
    prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]  # overround 0.9375
    snap = _snapshot("1.5", prices)

    def get_prices(market_id):
        return snap if market_id == "1.5" else None
    opp = scan_market(get_prices, "1.5", event_name="Event")
    assert opp is not None
    assert opp.event_name == "Event"
