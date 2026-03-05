from datetime import datetime, timezone
from decimal import Decimal

from core.cross_market_scanner import (
    scan_cross_market_btts,
    scan_cross_market_cs_mo,
    scan_cross_market_ou25,
)
from core.types import PriceSnapshot, SelectionPrice


def _sel(selection_id: str, name: str, back: str, lay: str, back_liq: str = "500", lay_liq: str = "500") -> SelectionPrice:
    return SelectionPrice(
        selection_id=selection_id,
        name=name,
        best_back_price=Decimal(back),
        available_to_back=Decimal(back_liq),
        best_lay_price=Decimal(lay),
        available_to_lay=Decimal(lay_liq),
    )


def _snap(market_id: str, selections: tuple[SelectionPrice, ...]) -> PriceSnapshot:
    return PriceSnapshot(
        market_id=market_id,
        selections=selections,
        timestamp=datetime.now(timezone.utc),
    )


def test_scan_cross_market_ou25_disabled_returns_none() -> None:
    mo = _snap(
        "1.100",
        (
            _sel("h", "Home", "6.0", "6.2"),
            _sel("d", "Draw", "6.0", "6.2"),
            _sel("a", "Away", "6.0", "6.2"),
        ),
    )
    ou = _snap(
        "1.200",
        (
            _sel("u25", "Under 2.5 Goals", "6.0", "6.2"),
            _sel("o25", "Over 2.5 Goals", "6.0", "6.2"),
        ),
    )

    assert scan_cross_market_ou25(mo, ou, event_name="X vs Y") is None


def test_scan_cross_market_btts_disabled_returns_none() -> None:
    mo = _snap(
        "1.101",
        (
            _sel("h", "Home", "6.0", "6.2"),
            _sel("d", "Draw", "6.0", "6.2"),
            _sel("a", "Away", "6.0", "6.2"),
        ),
    )
    btts = _snap(
        "1.201",
        (
            _sel("yes", "Both Teams To Score? Yes", "6.0", "6.2"),
            _sel("no", "Both Teams To Score? No", "6.0", "6.2"),
        ),
    )

    assert scan_cross_market_btts(mo, btts, event_name="A vs B") is None


def test_scan_cross_market_ou25_requires_named_selections() -> None:
    mo = _snap(
        "1.102",
        (
            _sel("h", "Home", "6.0", "6.2"),
            _sel("d", "Draw", "6.0", "6.2"),
            _sel("a", "Away", "6.0", "6.2"),
        ),
    )
    ou = _snap(
        "1.202",
        (
            _sel("x", "Over line wrong", "6.0", "6.2"),
            _sel("y", "Under line wrong", "6.0", "6.2"),
        ),
    )

    assert scan_cross_market_ou25(mo, ou, event_name="X vs Y") is None


def test_scan_cross_market_cs_mo_is_noop() -> None:
    cs = _snap(
        "1.301",
        (
            _sel("cs1", "0 - 0", "7.0", "7.2"),
            _sel("cs2", "1 - 0", "8.0", "8.2"),
            _sel("cs3", "0 - 1", "8.0", "8.2"),
        ),
    )
    mo = _snap(
        "1.101",
        (
            _sel("h", "Home", "2.2", "2.24"),
            _sel("d", "Draw", "3.4", "3.45"),
            _sel("a", "Away", "3.1", "3.15"),
        ),
    )
    assert scan_cross_market_cs_mo(cs, mo, event_name="A vs B") is None
