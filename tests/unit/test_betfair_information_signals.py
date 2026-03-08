from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from betfair.signals.event_linker import EventLinker
from betfair.signals.polymarket_adapter import PolymarketAdapter
from betfair.strategies.suspension_lag import evaluate_suspension_lag
from core.types import PriceSnapshot, SelectionPrice


def test_event_linker_matches_polymarket_to_betfair_event():
    linker = EventLinker(min_confidence=0.7)
    betfair_events = linker.build_betfair_events(
        {
            "1.1": {
                "event_id": "evt-1",
                "event_name": "Real Madrid vs Barcelona",
                "sport_name": "soccer",
                "competition_name": "La Liga",
                "market_start": "2026-03-08T20:00:00Z",
            }
        }
    )
    matches = linker.match_events(
        source="polymarket",
        external_events=[
            {
                "event_key": "real-madrid-vs-barcelona",
                "title": "Real Madrid vs Barcelona",
                "sport": "soccer",
                "competition": "La Liga",
                "scheduled_start": "2026-03-08T20:05:00Z",
            }
        ],
        betfair_events=betfair_events,
    )
    assert len(matches) == 1
    assert matches[0].betfair_event_id == "evt-1"


def test_event_linker_can_match_selection_style_market():
    linker = EventLinker(min_confidence=0.6)
    betfair_events = linker.build_betfair_events(
        {
            "1.2": {
                "event_id": "evt-2",
                "event_name": "World Cup 2026 Qualifying",
                "sport_name": "soccer",
                "competition_name": "FIFA World Cup",
                "market_start": "2026-03-10T20:00:00Z",
                "market_name": "To Qualify",
                "runner_names": ["Italy", "Netherlands", "Belgium"],
            }
        }
    )
    matches = linker.match_events(
        source="polymarket",
        external_events=[
            {
                "event_key": "italy-qualify-world-cup",
                "title": "Will Italy qualify for the 2026 FIFA World Cup?",
                "sport": "soccer",
                "competition": "FIFA World Cup",
                "scheduled_start": "2026-03-10T20:00:00Z",
            }
        ],
        betfair_events=betfair_events,
    )
    assert len(matches) == 1
    assert matches[0].betfair_event_id == "evt-2"


def test_polymarket_adapter_parses_market_rows():
    rows = [
        {
            "eventSlug": "real-madrid-vs-barcelona",
            "slug": "real-madrid-vs-barcelona-moneyline",
            "sportsMarketType": "soccer",
            "question": "Real Madrid vs Barcelona",
            "gameStartTime": "2026-03-08T20:00:00Z",
            "lastTradePrice": 0.64,
            "bestBid": 0.63,
            "bestAsk": 0.65,
            "league": "La Liga",
        }
    ]
    events = PolymarketAdapter._parse_market_rows(rows)
    assert len(events) == 1
    assert events[0].sport == "soccer"
    assert events[0].last_trade_price == 0.64


def test_suspension_lag_candidate_uses_polymarket_confirmation():
    snapshot = PriceSnapshot(
        market_id="1.1",
        selections=(
            SelectionPrice(
                selection_id="100",
                name="Real Madrid",
                best_back_price=Decimal("1.90"),
                available_to_back=Decimal("150"),
                best_lay_price=Decimal("1.95"),
                available_to_lay=Decimal("120"),
            ),
            SelectionPrice(
                selection_id="101",
                name="Barcelona",
                best_back_price=Decimal("4.10"),
                available_to_back=Decimal("90"),
                best_lay_price=Decimal("4.20"),
                available_to_lay=Decimal("90"),
            ),
        ),
        timestamp=datetime.now(timezone.utc),
        market_status="OPEN",
    )
    candidate = evaluate_suspension_lag(
        matched_event={
            "external_event_key": "real-madrid-vs-barcelona",
            "probability": 0.72,
            "match_confidence": 0.92,
            "quote_freshness_sec": 0.0,
        },
        snapshot=snapshot,
        market_meta={
            "event_name": "Real Madrid vs Barcelona",
            "market_type": "MATCH_ODDS",
            "competition_name": "La Liga",
            "sport_name": "soccer",
            "market_start": "2026-03-08T19:59:00Z",
        },
    )
    assert candidate is not None
    assert candidate["strategy_id"] == "betfair_suspension_lag"
    assert candidate["polymarket_confirmed"] is True
