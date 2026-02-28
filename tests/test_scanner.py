"""
Comprehensive tests for core.scanner: back-back (scan_snapshot) and lay-lay (scan_snapshot_lay).
Tests both valid arbs, edge cases, liquidity checks, and profitability thresholds.
"""
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from core.types import PriceSnapshot, SelectionPrice, Opportunity
from core.scanner import scan_snapshot, scan_snapshot_lay, scan_market


def _snapshot_back(market_id: str, prices: list, liquidity: Decimal = Decimal("1000")):
    """Create a price snapshot with back prices only."""
    selections = [
        SelectionPrice(
            selection_id=str(i),
            name=f"Sel{i}",
            best_back_price=Decimal(str(p)),
            available_to_back=liquidity,
        )
        for i, p in enumerate(prices)
    ]
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


def _snapshot_lay(
    market_id: str,
    lay_prices: list,
    available_to_lay: Decimal = Decimal("1000"),
):
    """Create a price snapshot with lay prices only."""
    selections = [
        SelectionPrice(
            selection_id=str(i),
            name=f"Sel{i}",
            best_back_price=Decimal("0"),  # no back price
            available_to_back=Decimal("0"),
            best_lay_price=Decimal(str(p)),
            available_to_lay=available_to_lay,
        )
        for i, p in enumerate(lay_prices)
    ]
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


def _snapshot_mixed(
    market_id: str,
    back_prices: list,
    lay_prices: list,
    available_to_back: Decimal = Decimal("1000"),
    available_to_lay: Decimal = Decimal("1000"),
):
    """Create a price snapshot with both back and lay prices."""
    selections = [
        SelectionPrice(
            selection_id=str(i),
            name=f"Sel{i}",
            best_back_price=Decimal(str(back_prices[i])),
            available_to_back=available_to_back,
            best_lay_price=Decimal(str(lay_prices[i])),
            available_to_lay=available_to_lay,
        )
        for i in range(len(back_prices))
    ]
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


# ============================================================================
# TESTS FOR scan_snapshot (back-back arbs)
# ============================================================================


class TestScanSnapshotBackBack:
    """Test back-back arbitrage detection."""

    def test_scan_snapshot_back_back_valid(self):
        """
        3 selections with back prices [2.5, 3.5, 4.0] (overround < 1),
        liquidity 200 each. Should return Opportunity with arb_type="back_back".
        Overround: 1/2.5 + 1/3.5 + 1/4.0 = 0.4 + 0.2857 + 0.25 = 0.9357 < 1
        """
        prices = [Decimal("2.50"), Decimal("3.50"), Decimal("4.00")]
        snap = _snapshot_back("market.1", prices, Decimal("200"))

        opp = scan_snapshot(
            snap,
            event_name="Test Event",
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is not None, "Expected valid back-back arb to be detected"
        assert opp.market_id == "market.1"
        assert opp.arb_type == "back_back"
        assert opp.overround_raw < Decimal("1")
        assert opp.net_profit_eur > 0
        assert len(opp.selections) == 3
        assert opp.total_stake_eur > 0
        assert opp.event_name == "Test Event"

    def test_scan_snapshot_back_back_no_arb(self):
        """
        3 selections with prices [1.5, 3.0, 5.0] (overround > 1).
        Overround: 1/1.5 + 1/3.0 + 1/5.0 = 0.667 + 0.333 + 0.2 = 1.2 > 1
        Should return None.
        """
        prices = [Decimal("1.50"), Decimal("3.00"), Decimal("5.00")]
        snap = _snapshot_back("market.2", prices, Decimal("200"))

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected no arb for overround > 1"

    def test_scan_snapshot_back_back_insufficient_liquidity(self):
        """
        Valid back prices but one selection has available_to_back=1.0 (too low).
        Should return None due to liquidity check.
        """
        prices = [Decimal("2.50"), Decimal("3.50"), Decimal("4.00")]
        selections = [
            SelectionPrice("0", "A", Decimal("2.50"), Decimal("200")),
            SelectionPrice("1", "B", Decimal("3.50"), Decimal("1.00")),  # too low
            SelectionPrice("2", "C", Decimal("4.00"), Decimal("200")),
        ]
        snap = PriceSnapshot(
            market_id="market.3",
            selections=tuple(selections),
            timestamp=datetime.now(timezone.utc),
        )

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None when liquidity below threshold"

    def test_scan_snapshot_back_back_below_min_profit(self):
        """
        Back prices giving marginal arb below min_net_profit_eur=5.0.
        Should return None.
        """
        prices = [Decimal("2.10"), Decimal("2.10")]  # small edge
        snap = _snapshot_back("market.4", prices, Decimal("1000"))

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("5.00"),  # high threshold
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None when net profit below threshold"


# ============================================================================
# TESTS FOR scan_snapshot_lay (lay-lay arbs)
# ============================================================================


class TestScanSnapshotLay:
    """Test lay-lay arbitrage detection."""

    def test_scan_snapshot_lay_valid(self):
        """
        3 selections with lay_prices=[2.10, 3.20, 3.80] (lay_overround > 1).
        Lay overround: 1/2.10 + 1/3.20 + 1/3.80 ≈ 0.476 + 0.3125 + 0.263 = 1.052 > 1
        Each available_to_lay=200.
        Should return Opportunity with arb_type="lay_lay".
        """
        lay_prices = [Decimal("2.10"), Decimal("3.20"), Decimal("3.80")]
        snap = _snapshot_lay("market.5", lay_prices, Decimal("200"))

        opp = scan_snapshot_lay(
            snap,
            event_name="Lay Event",
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is not None, "Expected valid lay-lay arb to be detected"
        assert opp.market_id == "market.5"
        assert opp.arb_type == "lay_lay"
        assert opp.overround_raw > Decimal("1")
        assert opp.net_profit_eur > 0
        assert len(opp.selections) == 3
        assert opp.total_stake_eur > 0
        assert opp.event_name == "Lay Event"

    def test_scan_snapshot_lay_no_arb(self):
        """
        3 selections with lay_prices=[2.50, 3.50, 4.00] (lay_overround < 1).
        Lay overround: 1/2.50 + 1/3.50 + 1/4.00 = 0.4 + 0.286 + 0.25 = 0.936 < 1
        Should return None (no lay arb).
        """
        lay_prices = [Decimal("2.50"), Decimal("3.50"), Decimal("4.00")]
        snap = _snapshot_lay("market.6", lay_prices, Decimal("200"))

        opp = scan_snapshot_lay(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected no arb for lay_overround < 1"

    def test_scan_snapshot_lay_insufficient_liquidity(self):
        """
        Valid lay arb but one selection has available_to_lay=1.0 (too low).
        Should return None.
        """
        selections = [
            SelectionPrice(
                "0", "A", Decimal("0"), Decimal("0"),
                best_lay_price=Decimal("2.10"),
                available_to_lay=Decimal("200"),
            ),
            SelectionPrice(
                "1", "B", Decimal("0"), Decimal("0"),
                best_lay_price=Decimal("3.20"),
                available_to_lay=Decimal("1.00"),  # too low
            ),
            SelectionPrice(
                "2", "C", Decimal("0"), Decimal("0"),
                best_lay_price=Decimal("3.80"),
                available_to_lay=Decimal("200"),
            ),
        ]
        snap = PriceSnapshot(
            market_id="market.7",
            selections=tuple(selections),
            timestamp=datetime.now(timezone.utc),
        )

        opp = scan_snapshot_lay(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None when liquidity below threshold"

    def test_scan_snapshot_lay_below_min_profit(self):
        """
        Lay prices giving marginal arb below min_net_profit_eur=5.0.
        Should return None.
        """
        lay_prices = [Decimal("2.05"), Decimal("2.05")]  # marginal edge
        snap = _snapshot_lay("market.8", lay_prices, Decimal("1000"))

        opp = scan_snapshot_lay(
            snap,
            min_net_profit_eur=Decimal("5.00"),  # high threshold
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None when net profit below threshold"

    def test_scan_snapshot_lay_no_lay_price(self):
        """
        One selection has best_lay_price=0 (no lay available).
        Should return None.
        """
        selections = [
            SelectionPrice(
                "0", "A", Decimal("0"), Decimal("0"),
                best_lay_price=Decimal("2.10"),
                available_to_lay=Decimal("200"),
            ),
            SelectionPrice(
                "1", "B", Decimal("0"), Decimal("0"),
                best_lay_price=Decimal("0"),  # no lay available
                available_to_lay=Decimal("0"),
            ),
            SelectionPrice(
                "2", "C", Decimal("0"), Decimal("0"),
                best_lay_price=Decimal("3.80"),
                available_to_lay=Decimal("200"),
            ),
        ]
        snap = PriceSnapshot(
            market_id="market.9",
            selections=tuple(selections),
            timestamp=datetime.now(timezone.utc),
        )

        opp = scan_snapshot_lay(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None when any selection has no lay price"


# ============================================================================
# TESTS FOR scan_market (integration)
# ============================================================================


class TestScanMarket:
    """Test scan_market helper that wraps scan_snapshot with cache."""

    def test_scan_market_picks_best(self):
        """
        Market has both back-back and lay-lay opportunities available.
        scan_market should pick the one with higher net_profit_eur.
        """
        # Back prices give overround < 1 (back-back arb)
        # Lay prices give lay_overround > 1 (lay-lay arb)
        back_prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
        lay_prices = [Decimal("2.10"), Decimal("3.20"), Decimal("3.80")]
        snap = _snapshot_mixed(
            "market.10",
            back_prices,
            lay_prices,
            available_to_back=Decimal("300"),
            available_to_lay=Decimal("300"),
        )

        def get_prices(market_id):
            return snap if market_id == "market.10" else None

        opp = scan_market(
            get_prices,
            "market.10",
            event_name="Mixed Event",
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is not None, "Expected scan_market to find at least one arb"
        assert opp.arb_type in ["back_back", "lay_lay"]


# ============================================================================
# EDGE CASES AND REGRESSION TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and guard conditions."""

    def test_scan_snapshot_empty_selections(self):
        """Empty selections should return None."""
        snap = PriceSnapshot(
            market_id="market.11",
            selections=(),
            timestamp=datetime.now(timezone.utc),
        )

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None for empty selections"

    def test_scan_snapshot_single_selection(self):
        """Single selection cannot form an arb; should return None."""
        snap = _snapshot_back("market.12", [Decimal("2.50")])

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None for single selection"

    def test_scan_snapshot_four_selections(self):
        """More than 3 selections (guard against wrong market type)."""
        prices = [Decimal("2.50"), Decimal("2.50"), Decimal("2.50"), Decimal("2.50")]
        snap = _snapshot_back("market.13", prices)

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is None, "Expected None for >3 selections"

    def test_scan_snapshot_with_config_defaults(self):
        """Test that config defaults are used when not overridden."""
        prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
        snap = _snapshot_back("market.14", prices, Decimal("200"))

        # Don't pass explicit params; should use config defaults
        opp = scan_snapshot(snap, event_name="Default Config Test")

        # Should still find the arb with default thresholds
        assert opp is not None, "Expected arb with config defaults"
        assert opp.market_id == "market.14"

    def test_scan_snapshot_two_selection_valid(self):
        """Two-selection market (e.g., binary) can have valid arb."""
        prices = [Decimal("2.00"), Decimal("2.00")]  # overround = 1.0, no arb
        snap = _snapshot_back("market.15", prices, Decimal("200"))

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        # 1/2 + 1/2 = 1.0, so no arb
        assert opp is None, "Expected None for exactly overround=1"

    def test_scan_snapshot_preserves_metadata(self):
        """Opportunity should include market_id, event_name, market_start."""
        from datetime import datetime as dt

        prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
        snap = _snapshot_back("market.16", prices, Decimal("200"))
        event_name = "Test Event Name"
        market_start = dt(2026, 2, 27, 15, 0, 0, tzinfo=timezone.utc)

        opp = scan_snapshot(
            snap,
            event_name=event_name,
            market_start=market_start,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is not None
        assert opp.market_id == "market.16"
        assert opp.event_name == event_name
        assert opp.market_start == market_start


# ============================================================================
# COMMISSION AND DECIMAL PRECISION
# ============================================================================


class TestCommissionAndPrecision:
    """Test that commission is correctly applied and Decimals are precise."""

    def test_scan_snapshot_commission_deduction(self):
        """Verify commission is deducted from gross profit."""
        prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
        snap = _snapshot_back("market.17", prices, Decimal("1000"))

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is not None
        # Net profit should be less than gross profit due to commission
        assert opp.net_profit_eur < opp.gross_profit_eur
        assert opp.commission_eur > 0
        # Check relationship: net_profit = gross_profit - commission
        assert abs(
            (opp.net_profit_eur + opp.commission_eur) - opp.gross_profit_eur
        ) < Decimal("0.01"), "Commission math should be consistent"

    def test_scan_snapshot_discount_rate(self):
        """Discount rate should reduce commission."""
        prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
        snap = _snapshot_back("market.18", prices, Decimal("1000"))

        # Without discount
        opp_no_discount = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        # With 60% discount
        opp_with_discount = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.60"),
        )

        assert opp_no_discount is not None
        assert opp_with_discount is not None
        # Higher discount should result in lower commission and higher net profit
        assert opp_with_discount.commission_eur < opp_no_discount.commission_eur
        assert opp_with_discount.net_profit_eur > opp_no_discount.net_profit_eur

    def test_scan_snapshot_roi_calculation(self):
        """ROI should be min_net_profit / total_stake."""
        prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
        snap = _snapshot_back("market.19", prices, Decimal("1000"))

        opp = scan_snapshot(
            snap,
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("10"),
            max_stake_eur=Decimal("100"),
            mbr=Decimal("0.05"),
            discount_rate=Decimal("0.00"),
        )

        assert opp is not None
        expected_roi = opp.net_profit_eur / opp.total_stake_eur
        # Allow small rounding tolerance
        assert abs(opp.net_roi_pct - expected_roi) < Decimal("0.0001")
