from __future__ import annotations

from portfolio.types import ModelShadowAccount
from polymarket.clob_client import PolymarketClobClient
from polymarket.features import build_feature_rows
from polymarket.gamma_client import PolymarketGammaClient
from polymarket.labels import QuantumFoldLabelStore
from polymarket.model_league import QuantumFoldModelLeague
from polymarket.paper_executor import PolymarketPaperExecutor


def test_gamma_client_normalizes_binary_sports_market():
    rows = [
        {
            "id": "evt-1",
            "slug": "real-madrid-vs-barcelona",
            "title": "Real Madrid vs Barcelona",
            "tags": [{"slug": "sports"}, {"slug": "soccer"}],
            "league": "La Liga",
            "startDate": "2026-03-08T20:00:00Z",
            "markets": [
                {
                    "id": "mkt-1",
                    "slug": "rm-v-barca-moneyline",
                    "question": "Real Madrid vs Barcelona",
                    "clobTokenIds": '["tok-rm","tok-barca"]',
                    "outcomes": '["Real Madrid","Barcelona"]',
                    "outcomePrices": '["0.62","0.38"]',
                    "bestBid": 0.61,
                    "bestAsk": 0.63,
                    "liquidityNum": 12000,
                    "volume24hrClob": 9000,
                }
            ],
        }
    ]

    snapshot = PolymarketGammaClient.normalize_events(rows, sports_filter={"soccer"})

    assert snapshot["event_count"] == 1
    assert snapshot["market_count"] == 1
    assert snapshot["token_count"] == 2
    assert snapshot["tokens"][0]["token_id"] == "tok-rm"
    assert snapshot["tokens"][1]["outcome"] == "Barcelona"
    assert snapshot["tokens"][1]["gamma_price"] == 0.38


def test_clob_client_parses_books_and_applies_best_bid_ask_updates():
    books = PolymarketClobClient.parse_order_books(
        [
            {
                "market": "mkt-1",
                "asset_id": "tok-rm",
                "timestamp": "1710000000000",
                "bids": [{"price": "0.61", "size": "150"}],
                "asks": [{"price": "0.63", "size": "120"}],
                "last_trade_price": "0.62",
                "tick_size": "0.01",
            }
        ]
    )

    assert books["tok-rm"]["midpoint"] == 0.62
    assert books["tok-rm"]["top_bid_size"] == 150.0

    PolymarketClobClient.apply_ws_message(
        books,
        {
            "event_type": "best_bid_ask",
            "asset_id": "tok-rm",
            "market": "mkt-1",
            "best_bid": "0.625",
            "best_ask": "0.635",
            "timestamp": "1710000001000",
        },
    )

    assert books["tok-rm"]["best_bid"] == 0.625
    assert books["tok-rm"]["midpoint"] == 0.63
    assert books["tok-rm"]["source"] == "clob_ws"


def test_clob_client_fetch_books_posts_token_ids(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "market": "mkt-1",
                    "asset_id": "tok-rm",
                    "timestamp": "1710000000000",
                    "bids": [{"price": "0.61", "size": "150"}],
                    "asks": [{"price": "0.63", "size": "120"}],
                    "last_trade_price": "0.62",
                }
            ]

    class _Client:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return _Response()

    monkeypatch.setattr("polymarket.clob_client.httpx.Client", _Client)

    client = PolymarketClobClient(base_url="https://clob.polymarket.com")
    books = client.fetch_books(["tok-rm", "tok-barca"])

    assert captured["url"] == "https://clob.polymarket.com/books"
    assert captured["json"] == [{"token_id": "tok-rm"}, {"token_id": "tok-barca"}]
    assert books["tok-rm"]["midpoint"] == 0.62


def test_feature_label_and_model_flow(tmp_path):
    token_rows = [
        {
            "token_id": "tok-rm",
            "market_slug": "rm-v-barca-moneyline",
            "event_slug": "real-madrid-vs-barcelona",
            "title": "Real Madrid vs Barcelona",
            "sport": "soccer",
            "competition": "La Liga",
            "gamma_price": 0.62,
            "midpoint": 0.63,
            "last_trade_price": 0.631,
            "best_bid": 0.62,
            "best_ask": 0.64,
            "bid_depth": 3200.0,
            "ask_depth": 2800.0,
            "asks": [{"price": 0.64, "size": 200.0}],
            "bids": [{"price": 0.62, "size": 180.0}],
            "book_timestamp": "2026-03-08T10:00:00Z",
        },
        {
            "token_id": "tok-barca",
            "market_slug": "rm-v-barca-moneyline",
            "event_slug": "real-madrid-vs-barcelona",
            "title": "Real Madrid vs Barcelona",
            "sport": "soccer",
            "competition": "La Liga",
            "gamma_price": 0.38,
            "midpoint": 0.37,
            "last_trade_price": 0.369,
            "best_bid": 0.36,
            "best_ask": 0.38,
            "bid_depth": 2500.0,
            "ask_depth": 3000.0,
            "asks": [{"price": 0.38, "size": 180.0}],
            "bids": [{"price": 0.36, "size": 160.0}],
            "book_timestamp": "2026-03-08T10:00:00Z",
        },
    ]
    histories = {
        "tok-rm": [
            {"ts": "2026-03-08T09:58:00Z", "price": 0.60},
            {"ts": "2026-03-08T09:59:00Z", "price": 0.615},
            {"ts": "2026-03-08T10:00:00Z", "price": 0.63},
        ],
        "tok-barca": [
            {"ts": "2026-03-08T09:58:00Z", "price": 0.40},
            {"ts": "2026-03-08T09:59:00Z", "price": 0.385},
            {"ts": "2026-03-08T10:00:00Z", "price": 0.37},
        ],
    }

    features = build_feature_rows(
        token_rows,
        histories=histories,
        stale_quote_seconds=30,
        fee_bps=20.0,
        queue_penalty_bps=8.0,
    )
    row = next(item for item in features if item["token_id"] == "tok-rm")
    assert row["coherence_score"] > 0
    assert row["folding_confidence"] > 0
    assert row["spread_bps"] > 0

    league = QuantumFoldModelLeague("polymarket_quantum_fold", starting_balance=1000.0)
    predictions = league.track_example(row)
    store = QuantumFoldLabelStore(tmp_path, horizons=[120])
    store.track_examples(
        [
            {
                "example_id": "tok-rm:1",
                "token_id": "tok-rm",
                "market_slug": row["market_slug"],
                "event_slug": row["event_slug"],
                "tracked_at": "2026-03-08T09:55:00Z",
                "entry_midpoint": row["midpoint"],
                "cost_buffer": row["cost_buffer"],
                "features": {
                    "coherence_score": row["coherence_score"],
                    "folding_confidence": row["folding_confidence"],
                    "orderbook_imbalance": row["orderbook_imbalance"],
                    "momentum_bias": row["momentum_bias"],
                    "energy_proxy": row["energy_proxy"],
                    "spread_bps": row["spread_bps"],
                    "gamma_delta": row["gamma_delta"],
                    "return_2m": row["return_2m"],
                    "basin_depth": row["basin_depth"],
                    "relaxation_speed": row["relaxation_speed"],
                },
                "model_predictions": predictions,
            }
        ]
    )

    result = store.update_labels(
        {
            "tok-rm": {
                "midpoint": 0.71,
                "resolved": True,
                "closed": True,
                "resolution": 1.0,
            }
        }
    )
    assert result["completed"] == 2

    league.settle_labels(result["settled_labels"], primary_horizon=120)
    accounts = league.build_accounts()
    hybrid = next(item for item in accounts if item.model_id == "hybrid_transition")
    assert hybrid.settled_count >= 2
    assert "recent_learning_brier_lift" in hybrid.metrics


def test_paper_executor_models_fill_and_realized_pnl():
    executor = PolymarketPaperExecutor(
        starting_balance=1000.0,
        fee_bps=20.0,
        queue_penalty_bps=8.0,
        max_open_positions=5,
        max_notional_per_trade=150.0,
        max_positions_per_event=1,
        drawdown_halt_pct=10.0,
    )
    feature_row = {
        "token_id": "tok-rm",
        "market_slug": "rm-v-barca-moneyline",
        "event_slug": "real-madrid-vs-barcelona",
        "title": "Real Madrid vs Barcelona",
        "outcome": "Real Madrid",
        "sport": "soccer",
        "competition": "La Liga",
        "best_ask": 0.60,
        "asks": [
            {"price": 0.60, "size": 50.0},
            {"price": 0.62, "size": 200.0},
        ],
        "tick_size": 0.01,
        "folding_confidence": 0.77,
        "coherence_score": 0.72,
    }
    trade = executor.open_trade(feature_row, score_probability=0.61, notional_usd=60.0)

    assert trade["entry_price"] > 0.60
    assert trade["quantity"] > 0

    closed = executor.close_trade(
        trade,
        {
            "best_bid": 0.68,
            "bids": [{"price": 0.68, "size": 200.0}],
            "tick_size": 0.01,
        },
        reason="test_exit",
    )

    assert closed["status"] == "CLOSED"
    assert closed["net_pnl_usd"] > 0
