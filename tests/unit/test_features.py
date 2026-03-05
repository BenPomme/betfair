from datetime import datetime, timedelta, timezone
from decimal import Decimal

from core.types import Opportunity, PriceSnapshot, SelectionPrice
from strategy.features import build_feature_vector


def _snapshot() -> PriceSnapshot:
    return PriceSnapshot(
        market_id="1.test",
        selections=(
            SelectionPrice("1", "A", Decimal("3.20"), Decimal("120"), Decimal("3.25"), Decimal("90")),
            SelectionPrice("2", "B", Decimal("3.20"), Decimal("110"), Decimal("3.30"), Decimal("85")),
            SelectionPrice("3", "C", Decimal("3.20"), Decimal("100"), Decimal("3.35"), Decimal("80")),
        ),
        timestamp=datetime.now(timezone.utc),
    )


def _opportunity() -> Opportunity:
    return Opportunity(
        market_id="1.test",
        event_name="Event",
        market_start=datetime.now(timezone.utc) + timedelta(minutes=30),
        arb_type="back_back",
        selections=(
            {"selection_id": "1", "name": "A", "back_price": 3.2, "stake_eur": 33.33, "liquidity_eur": 120},
            {"selection_id": "2", "name": "B", "back_price": 3.2, "stake_eur": 33.33, "liquidity_eur": 110},
            {"selection_id": "3", "name": "C", "back_price": 3.2, "stake_eur": 33.34, "liquidity_eur": 100},
        ),
        total_stake_eur=Decimal("100"),
        overround_raw=Decimal("0.9375"),
        gross_profit_eur=Decimal("6.66"),
        commission_eur=Decimal("0.33"),
        net_profit_eur=Decimal("6.33"),
        net_roi_pct=Decimal("0.0633"),
        liquidity_by_selection=(Decimal("120"), Decimal("110"), Decimal("100")),
    )


def test_feature_vector_is_deterministic():
    snap = _snapshot()
    opp = _opportunity()
    f1 = build_feature_vector(snap, opp, market_start=opp.market_start, previous_snapshot=None)
    f2 = build_feature_vector(snap, opp, market_start=opp.market_start, previous_snapshot=None)
    assert f1 == f2
    assert f1.market_id == "1.test"
    assert f1.selection_count == 3
    assert f1.microstructure.in_play is False
    assert f1.microstructure.selection_count == 3
    assert f1.microstructure.weighted_spread > Decimal("0")
    assert f1.microstructure.lay_back_ratio > Decimal("0")
    assert f1.microstructure.top_of_book_concentration > Decimal("0")
