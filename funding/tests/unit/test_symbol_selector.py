from decimal import Decimal

import config
from funding.strategy.symbol_selector import (
    compute_book_metrics,
    qualify_symbol_for_trading,
    rank_qualified_opportunities,
)


def _book(bid_price: str, ask_price: str, bid_qty: str = "1000", ask_qty: str = "1000") -> dict:
    return {
        "bids": [[bid_price, bid_qty], [bid_price, bid_qty]],
        "asks": [[ask_price, ask_qty], [ask_price, ask_qty]],
    }


def test_compute_book_metrics_returns_spread_and_depth():
    metrics = compute_book_metrics(_book("100", "100.10"))
    assert metrics is not None
    assert metrics["bid_depth_usd"] > Decimal("0")
    assert metrics["spread_bps"] > Decimal("0")


def test_qualify_symbol_rejects_wide_spread(monkeypatch):
    monkeypatch.setattr(config, "FUNDING_MAX_SPREAD_BPS", Decimal("5"))
    ok, reason, _ = qualify_symbol_for_trading(
        "BTCUSDT",
        Decimal("1000"),
        Decimal("5"),
        compute_book_metrics(_book("100", "100.50")),
        compute_book_metrics(_book("100", "100.50")),
    )
    assert ok is False
    assert reason == "spread_too_wide"


def test_qualify_symbol_rejects_insufficient_depth(monkeypatch):
    monkeypatch.setattr(config, "FUNDING_MIN_DEPTH_USD", Decimal("1000000"))
    ok, reason, _ = qualify_symbol_for_trading(
        "BTCUSDT",
        Decimal("1000"),
        Decimal("5"),
        compute_book_metrics(_book("100", "100.01", bid_qty="1", ask_qty="1")),
        compute_book_metrics(_book("100", "100.01", bid_qty="1", ask_qty="1")),
    )
    assert ok is False
    assert reason == "depth_insufficient"


def test_rank_qualified_opportunities_prefers_higher_net_edge():
    ranked = rank_qualified_opportunities([
        {
            "opportunity": "A",
            "net_expected_edge_usd": Decimal("1"),
            "liquidity_score": Decimal("100"),
            "basis_bps": Decimal("3"),
            "combined_spread_bps": Decimal("5"),
        },
        {
            "opportunity": "B",
            "net_expected_edge_usd": Decimal("3"),
            "liquidity_score": Decimal("50"),
            "basis_bps": Decimal("2"),
            "combined_spread_bps": Decimal("4"),
        },
    ])
    assert ranked[0]["opportunity"] == "B"
