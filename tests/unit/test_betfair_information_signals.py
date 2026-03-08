from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import config
from betfair.models.polymarket_binary_ranker import PolymarketBinaryRanker
from betfair.signals.event_linker import EventLinker
from betfair.signals.polymarket_adapter import PolymarketAdapter
from betfair.signals.thesportsdb_adapter import TheSportsDBAdapter
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


def test_suspension_lag_upgrades_to_high_when_multi_source_confirmed():
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
            "source_mix": ["betfair_suspend_resume", "polymarket", "thesportsdb"],
            "external_source_count": 2,
            "polymarket_confirmed": True,
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
    assert candidate["event_confirmation_level"] == "high"
    assert "thesportsdb" in candidate["source_mix"]


def test_thesportsdb_adapter_normalizes_events():
    adapter = TheSportsDBAdapter()
    event = adapter._normalize_event(
        {
            "idEvent": "987",
            "strHomeTeam": "Real Madrid",
            "strAwayTeam": "Barcelona",
            "dateEvent": "2026-03-08",
            "strTime": "20:00:00",
            "strLeague": "La Liga",
        },
        "Soccer",
    )
    assert event["source"] == "thesportsdb"
    assert event["title"] == "Real Madrid vs Barcelona"
    assert event["sport"] == "soccer"


def test_polymarket_binary_ranker_labels_and_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    ranker = PolymarketBinaryRanker("betfair_core")
    candidate = {
        "candidate_id": "cand-1",
        "market_id": "rm-barca-binary",
        "event_key": "rm-barca",
        "reason": "momentum",
        "expected_edge": 0.05,
        "strategy_context": {
            "sport": "soccer",
            "price": 0.62,
            "liquidity": 6000.0,
            "spread_bps": 120.0,
        },
    }
    tracked = ranker.track_candidates([candidate])
    assert tracked["tracked"] == 1

    pending = ranker.load_pending()
    pending["rm-barca-binary"]["created_at"] = "2026-03-08T10:00:00Z"
    ranker.save_pending(pending)

    result = ranker.update_labels(
        current_quotes=[
            {
                "market_slug": "rm-barca-binary",
                "last_trade_price": 0.70,
            }
        ],
        min_elapsed_seconds=60,
    )
    assert result["completed"] == 1

    model = ranker.rebuild_model()
    assert model["labeled_examples"] == 1

    scored = ranker.score_candidate(candidate, model=model)
    assert scored["learned_count"] == 1
    assert isinstance(scored["bucket"], list)
