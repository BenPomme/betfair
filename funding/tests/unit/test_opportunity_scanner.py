"""Unit tests for funding opportunity scanner."""
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from funding.core.opportunity_scanner import proportional_position_size, scan_opportunities
from funding.core.schemas import FundingSnapshot


def _make_snapshot(
    symbol: str,
    funding_rate: str = "0.0003",
    mark_price: str = "50000",
    index_price: str = "50000",
) -> FundingSnapshot:
    return FundingSnapshot(
        symbol=symbol,
        funding_rate=Decimal(funding_rate),
        next_funding_time=datetime.now(timezone.utc) + timedelta(hours=4),
        mark_price=Decimal(mark_price),
        index_price=Decimal(index_price),
        open_interest=Decimal("1000"),
        timestamp=datetime.now(timezone.utc),
    )


class TestScanOpportunities:
    def test_high_funding_triggers(self):
        """High funding rate should produce an opportunity."""
        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0005")}
        volume = {"BTCUSDT": Decimal("100000000")}
        result = scan_opportunities(snapshots, volume)
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"
        assert result[0].annualized_yield > Decimal("0.10")

    def test_low_funding_rejected(self):
        """Funding rate below minimum should be filtered out."""
        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.00005")}
        volume = {"BTCUSDT": Decimal("100000000")}
        result = scan_opportunities(snapshots, volume)
        assert len(result) == 0

    def test_negative_funding_rejected(self):
        """Negative funding rate should be filtered out."""
        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "-0.0003")}
        volume = {"BTCUSDT": Decimal("100000000")}
        result = scan_opportunities(snapshots, volume)
        assert len(result) == 0

    def test_low_volume_rejected(self):
        """Volume below minimum should be filtered out."""
        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0005")}
        volume = {"BTCUSDT": Decimal("1000000")}  # $1M, well below $50M minimum
        result = scan_opportunities(snapshots, volume)
        assert len(result) == 0

    def test_sorted_by_yield(self):
        """Opportunities should be sorted by annualized yield descending."""
        snapshots = {
            "ETHUSDT": _make_snapshot("ETHUSDT", "0.0003"),
            "BTCUSDT": _make_snapshot("BTCUSDT", "0.0010"),
            "SOLUSDT": _make_snapshot("SOLUSDT", "0.0005"),
        }
        volume = {
            "ETHUSDT": Decimal("100000000"),
            "BTCUSDT": Decimal("100000000"),
            "SOLUSDT": Decimal("100000000"),
        }
        result = scan_opportunities(snapshots, volume)
        assert len(result) == 3
        assert result[0].symbol == "BTCUSDT"  # Highest rate
        assert result[1].symbol == "SOLUSDT"
        assert result[2].symbol == "ETHUSDT"  # Lowest rate

    def test_watchlist_filter(self):
        """Only symbols in watchlist should be scanned."""
        snapshots = {
            "BTCUSDT": _make_snapshot("BTCUSDT", "0.0005"),
            "ETHUSDT": _make_snapshot("ETHUSDT", "0.0005"),
        }
        volume = {
            "BTCUSDT": Decimal("100000000"),
            "ETHUSDT": Decimal("100000000"),
        }
        result = scan_opportunities(snapshots, volume, watchlist={"BTCUSDT"})
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"

    def test_position_size_capped(self):
        """Position size should respect max position config."""
        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0005")}
        volume = {"BTCUSDT": Decimal("100000000")}
        result = scan_opportunities(
            snapshots, volume, max_position=Decimal("200")
        )
        assert result[0].position_size == Decimal("200")

    def test_expected_funding_payment(self):
        """Expected funding payment should be calculated correctly."""
        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0003")}
        volume = {"BTCUSDT": Decimal("100000000")}
        result = scan_opportunities(
            snapshots, volume, max_position=Decimal("1000")
        )
        # $1000 × 0.03% = $0.30
        assert result[0].expected_funding_payment == Decimal("0.30")

    def test_empty_snapshots(self):
        """No snapshots should return empty list."""
        result = scan_opportunities({}, {})
        assert result == []

    def test_proportional_position_size_splits_remaining_budget(self):
        result = proportional_position_size(
            remaining_exposure=Decimal("50000"),
            remaining_slots=10,
            max_position=Decimal("5000"),
        )
        assert result == Decimal("5000.00")

    def test_proportional_position_size_respects_remaining_exposure(self):
        result = proportional_position_size(
            remaining_exposure=Decimal("3600"),
            remaining_slots=4,
            max_position=Decimal("5000"),
        )
        assert result == Decimal("900.00")
