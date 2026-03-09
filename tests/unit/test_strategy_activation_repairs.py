from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from betfair.strategies.crossbook_consensus import evaluate_crossbook_consensus
from betfair.strategies.polymarket_binary_research import build_polymarket_binary_candidates
from betfair.strategies.suspension_lag import evaluate_suspension_lag
from betfair.strategies.timezone_decay import evaluate_timezone_decay
from core.types import PriceSnapshot, SelectionPrice
from polymarket.engine import PolymarketQuantumFoldEngine
from polymarket.model_league import OnlineEdgeModel


def _snapshot(*, back_price: str = "1.70", lay_price: str = "1.82", available: str = "60") -> PriceSnapshot:
    return PriceSnapshot(
        market_id="1.234",
        timestamp=datetime.now(timezone.utc),
        market_status="OPEN",
        selections=(
            SelectionPrice(
                selection_id="101",
                name="Home",
                best_back_price=Decimal(back_price),
                best_lay_price=Decimal(lay_price),
                available_to_back=Decimal(available),
                available_to_lay=Decimal("80"),
            ),
            SelectionPrice(
                selection_id="102",
                name="Away",
                best_back_price=Decimal("2.40"),
                best_lay_price=Decimal("2.46"),
                available_to_back=Decimal("45"),
                available_to_lay=Decimal("45"),
            ),
        ),
    )


def test_crossbook_consensus_accepts_two_sources_with_confident_match():
    candidate = evaluate_crossbook_consensus(
        consensus_row={
            "event_key": "match-1",
            "source_count": 2,
            "consensus_prob": 0.62,
            "consensus_dispersion": 0.01,
            "sources": ["polymarket", "thesportsdb"],
        },
        snapshot=_snapshot(back_price="1.80"),
        market_meta={"event_name": "Match 1"},
        matched_event={"match_confidence": 0.82},
    )

    assert candidate is not None
    assert candidate["external_source_count"] == 2


def test_crossbook_consensus_accepts_single_quote_source_with_dual_source_event_confirmation():
    candidate = evaluate_crossbook_consensus(
        consensus_row={
            "event_key": "match-1",
            "source_count": 1,
            "consensus_prob": 0.62,
            "consensus_dispersion": 0.01,
            "sources": ["polymarket"],
        },
        snapshot=_snapshot(back_price="1.80"),
        market_meta={"event_name": "Match 1"},
        matched_event={
            "match_confidence": 0.84,
            "external_source_count": 2,
            "source_mix": ["polymarket", "thesportsdb"],
        },
    )

    assert candidate is not None
    assert candidate["external_source_count"] == 2
    assert "thesportsdb" in candidate["source_mix"]


def test_suspension_lag_accepts_multi_source_signal_below_old_floor():
    candidate = evaluate_suspension_lag(
        matched_event={
            "probability": 0.60,
            "source_mix": ["polymarket", "thesportsdb"],
            "external_source_count": 2,
            "match_confidence": 0.81,
        },
        snapshot=_snapshot(back_price="1.78"),
        market_meta={"event_name": "Match 1", "market_start": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()},
    )

    assert candidate is not None
    assert candidate["event_confirmation_level"] in {"medium", "high"}


def test_timezone_decay_skips_illiquid_board_and_returns_viable_candidate():
    snapshot = PriceSnapshot(
        market_id="1.345",
        timestamp=datetime.now(timezone.utc),
        market_status="OPEN",
        selections=(
            SelectionPrice(
                selection_id="201",
                name="Thin Favorite",
                best_back_price=Decimal("1.50"),
                best_lay_price=Decimal("1.90"),
                available_to_back=Decimal("5"),
                available_to_lay=Decimal("10"),
            ),
            SelectionPrice(
                selection_id="202",
                name="Viable Runner",
                best_back_price=Decimal("1.72"),
                best_lay_price=Decimal("1.80"),
                available_to_back=Decimal("75"),
                available_to_lay=Decimal("70"),
            ),
            SelectionPrice(
                selection_id="203",
                name="Away",
                best_back_price=Decimal("2.60"),
                best_lay_price=Decimal("2.66"),
                available_to_back=Decimal("35"),
                available_to_lay=Decimal("35"),
            ),
        ),
    )

    candidate = evaluate_timezone_decay(
        snapshot=snapshot,
        market_meta={"event_name": "Night Match", "market_start": "2026-03-09T23:30:00Z"},
        stale_snapshot_ratio=0.45,
    )

    assert candidate is not None
    assert candidate["selection_key"] == "202"


def test_polymarket_binary_research_relaxed_filters_yield_candidate():
    candidates = build_polymarket_binary_candidates(
        [
            {
                "title": "Will Team A win?",
                "event_slug": "team-a-win",
                "market_slug": "team-a-win",
                "probability": 0.52,
                "best_bid": 0.50,
                "best_ask": 0.53,
                "liquidity": 450,
                "one_day_price_change": 0.05,
                "one_week_price_change": 0.03,
                "sport": "soccer",
            }
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["reason"] in {"momentum", "flow_dislocation"}


def test_quantum_fold_shadow_trade_counts_use_baseline_probability():
    model = OnlineEdgeModel(
        model_id="hybrid_transition",
        feature_names=["coherence_score"],
        weights=[0.5],
        bias=0.0,
    )

    model.settle(
        prediction=0.73,
        features={"coherence_score": 0.8},
        target=1,
        net_return=0.04,
        baseline_probability=0.72,
    )
    model.settle(
        prediction=0.76,
        features={"coherence_score": 0.8},
        target=1,
        net_return=0.04,
        baseline_probability=0.72,
    )

    assert model.shadow_trade_count == 1


def test_quantum_fold_book_universe_prefers_midpriced_liquid_tokens(tmp_path):
    engine = PolymarketQuantumFoldEngine(tmp_path, initial_balance=7500.0)
    engine.book_universe_size = 4
    start_time = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()

    selected = engine._select_book_universe(
        [
            {
                "token_id": "extreme-yes",
                "title": "Real Madrid vs Barcelona",
                "market_slug": "real-madrid-vs-barcelona-moneyline",
                "event_slug": "real-madrid-vs-barcelona",
                "teams_or_players": ["Real Madrid", "Barcelona"],
                "start_time": start_time,
                "gamma_price": 0.001,
                "liquidity": 2_000_000,
                "volume_24hr": 50_000,
                "resolved": False,
                "closed": False,
            },
            {
                "token_id": "extreme-no",
                "title": "Real Madrid vs Barcelona",
                "market_slug": "real-madrid-vs-barcelona-moneyline",
                "event_slug": "real-madrid-vs-barcelona",
                "teams_or_players": ["Real Madrid", "Barcelona"],
                "start_time": start_time,
                "gamma_price": 0.999,
                "liquidity": 2_000_000,
                "volume_24hr": 50_000,
                "resolved": False,
                "closed": False,
            },
            {
                "token_id": "mid-1",
                "title": "Real Madrid vs Barcelona",
                "market_slug": "real-madrid-vs-barcelona-moneyline",
                "event_slug": "real-madrid-vs-barcelona",
                "teams_or_players": ["Real Madrid", "Barcelona"],
                "start_time": start_time,
                "gamma_price": 0.48,
                "liquidity": 8_000,
                "volume_24hr": 2_500,
                "resolved": False,
                "closed": False,
            },
            {
                "token_id": "mid-2",
                "title": "Real Madrid vs Barcelona",
                "market_slug": "real-madrid-vs-barcelona-moneyline",
                "event_slug": "real-madrid-vs-barcelona",
                "teams_or_players": ["Real Madrid", "Barcelona"],
                "start_time": start_time,
                "gamma_price": 0.56,
                "liquidity": 7_500,
                "volume_24hr": 2_000,
                "resolved": False,
                "closed": False,
            },
            {
                "token_id": "mid-3",
                "title": "Real Madrid vs Barcelona",
                "market_slug": "real-madrid-vs-barcelona-moneyline",
                "event_slug": "real-madrid-vs-barcelona",
                "teams_or_players": ["Real Madrid", "Barcelona"],
                "start_time": start_time,
                "gamma_price": 0.41,
                "liquidity": 6_000,
                "volume_24hr": 1_500,
                "resolved": False,
                "closed": False,
            },
            {
                "token_id": "mid-4",
                "title": "Real Madrid vs Barcelona",
                "market_slug": "real-madrid-vs-barcelona-moneyline",
                "event_slug": "real-madrid-vs-barcelona",
                "teams_or_players": ["Real Madrid", "Barcelona"],
                "start_time": start_time,
                "gamma_price": 0.63,
                "liquidity": 5_500,
                "volume_24hr": 1_400,
                "resolved": False,
                "closed": False,
            },
        ]
    )

    selected_ids = [row["token_id"] for row in selected]

    assert selected_ids == ["mid-1", "mid-2", "mid-3", "mid-4"]


def test_quantum_fold_blocks_long_horizon_outrights_and_allows_near_term_matchups(tmp_path):
    engine = PolymarketQuantumFoldEngine(tmp_path, initial_balance=7500.0)

    near_term_match = {
        "token_id": "match-1",
        "title": "Real Madrid vs Barcelona",
        "market_slug": "real-madrid-vs-barcelona-moneyline",
        "event_slug": "real-madrid-vs-barcelona",
        "teams_or_players": ["Real Madrid", "Barcelona"],
        "start_time": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
        "end_time": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
    }
    long_horizon_outright = {
        "token_id": "future-1",
        "title": "Will Brazil win the 2026 FIFA World Cup?",
        "market_slug": "will-brazil-win-the-2026-fifa-world-cup-183",
        "event_slug": "2026-fifa-world-cup-winner-595",
        "teams_or_players": ["2026 FIFA World Cup winner"],
        "start_time": (datetime.now(timezone.utc) - timedelta(days=250)).isoformat(),
        "end_time": (datetime.now(timezone.utc) + timedelta(days=400)).isoformat(),
    }
    already_started_outright = {
        "token_id": "future-2",
        "title": "Will Arsenal win the 2025-26 English Premier League?",
        "market_slug": "will-arsenal-win-the-202526-english-premier-league",
        "event_slug": "english-premier-league-winner",
        "teams_or_players": ["English Premier League winner"],
        "start_time": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        "end_time": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    }

    assert engine._is_short_horizon_market(near_term_match) is True
    assert engine._is_short_horizon_market(long_horizon_outright) is False
    assert engine._is_short_horizon_market(already_started_outright) is False


def test_quantum_fold_archives_old_runtime_when_short_horizon_policy_is_first_applied(tmp_path):
    runtime_dir = tmp_path / "pmqf"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "quantum_fold_examples.jsonl").write_text('{"example_id":"old"}\n', encoding="utf-8")
    (runtime_dir / "quantum_fold_pending_examples.json").write_text('{"old":{"example_id":"old"}}', encoding="utf-8")
    (runtime_dir / "summary_history.jsonl").write_text('{"status":"legacy"}\n', encoding="utf-8")

    engine = PolymarketQuantumFoldEngine(runtime_dir, initial_balance=7500.0)
    engine.start()
    engine.stop()

    archive_root = runtime_dir / "archive"
    archive_dirs = [path for path in archive_root.iterdir() if path.is_dir()]
    assert len(archive_dirs) == 1
    archived = archive_dirs[0]
    assert (archived / "quantum_fold_examples.jsonl").exists()
    assert (archived / "quantum_fold_pending_examples.json").exists()
    assert (archived / "summary_history.jsonl").exists()
    assert (archived / "archive_manifest.json").exists()
    assert (runtime_dir / "quantum_fold_policy_manifest.json").exists()
    assert not (runtime_dir / "quantum_fold_examples.jsonl").exists()
