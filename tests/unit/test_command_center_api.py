from __future__ import annotations

from fastapi.testclient import TestClient

import config
from monitoring import command_center
from portfolio.accounting import build_strategy_account
from portfolio.state_store import PortfolioStateStore
from portfolio.types import ModelShadowAccount


class _DummyManager:
    def status(self, portfolio_id: str):
        store = PortfolioStateStore(portfolio_id)
        return {"running": store.read_pid() is not None, "pid": store.read_pid(), "heartbeat": store.read_heartbeat()}


def test_command_center_portfolio_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    betfair_store = PortfolioStateStore("betfair_core")
    hedge_store = PortfolioStateStore("hedge_validation")

    betfair_store.write_account(
        build_strategy_account(
            portfolio_id="betfair_core",
            currency="EUR",
            starting_balance=1000.0,
            current_balance=1010.0,
            realized_pnl=10.0,
        )
    )
    betfair_store.write_state({"portfolio_id": "betfair_core", "running": True, "status": "running", "mode": "paper", "balance_eur": 1010.0})
    betfair_store.write_readiness({"status": "monitoring"})
    betfair_store.write_balance_history([{"ts": "2026-03-06T00:00:00Z", "balance": 1000.0}, {"ts": "2026-03-06T01:00:00Z", "balance": 1010.0}])
    betfair_store.write_pid(1234)

    hedge_store.write_account(
        build_strategy_account(
            portfolio_id="hedge_validation",
            currency="USD",
            starting_balance=50000.0,
            current_balance=50025.0,
            realized_pnl=25.0,
        )
    )
    hedge_store.write_state({
        "portfolio_id": "hedge_validation",
        "running": True,
        "status": "running",
        "mode": "paper",
        "trade_count": 0,
        "open_hedges": 0,
        "realized_net_pnl_usd": 25.0,
        "realized_roi_pct": 0.05,
    })
    hedge_store.write_readiness({"validation_ready": False, "blockers": ["closed_hedges_minimum"]})
    hedge_store.write_balance_history([{"ts": "2026-03-06T00:00:00Z", "balance": 50000.0}, {"ts": "2026-03-06T01:00:00Z", "balance": 50025.0}])
    hedge_store.write_pid(5678)

    client = TestClient(command_center.app)

    portfolios = client.get("/api/portfolios").json()["portfolios"]
    ids = {item["portfolio_id"] for item in portfolios}
    assert {"betfair_core", "hedge_validation"}.issubset(ids)

    betfair_state = client.get("/api/state").json()
    hedge_state = client.get("/api/funding/state").json()
    compare = client.get("/api/compare/portfolios").json()

    assert betfair_state["portfolio_id"] == "betfair_core"
    assert hedge_state["portfolio_id"] == "hedge_validation"
    assert compare["series"]["betfair_core"][-1]["balance"] == 1010.0
    assert compare["series"]["hedge_validation"][-1]["balance"] == 50025.0


def test_command_center_does_not_report_stale_runner_as_live(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    hedge_store = PortfolioStateStore("hedge_validation")
    hedge_store.write_account(
        build_strategy_account(
            portfolio_id="hedge_validation",
            currency="USD",
            starting_balance=50000.0,
            current_balance=50000.0,
            realized_pnl=0.0,
        )
    )
    hedge_store.write_state(
        {
            "portfolio_id": "hedge_validation",
            "running": True,
            "status": "running",
            "mode": "paper",
        }
    )

    client = TestClient(command_center.app)
    summary = client.get("/api/portfolios/hedge_validation/summary").json()

    assert summary["running"] is False
    assert summary["status"] == "idle"


def test_command_center_notification_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    client = TestClient(command_center.app)
    state = client.get("/api/notifications/state").json()

    assert "discord_configured" in state
    assert "notification_failures" in state


def test_command_center_exposes_polymarket_quantum_fold_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    store = PortfolioStateStore("polymarket_quantum_fold")
    store.write_account(
        build_strategy_account(
            portfolio_id="polymarket_quantum_fold",
            currency="USD",
            starting_balance=7500.0,
            current_balance=7525.0,
            realized_pnl=25.0,
            trade_count=12,
        )
    )
    store.write_state(
        {
            "portfolio_id": "polymarket_quantum_fold",
            "running": True,
            "status": "running",
            "mode": "paper",
            "realized_pnl_usd": 25.0,
            "current_balance_usd": 7525.0,
            "trade_count": 12,
            "source_health": {
                "gamma": {"healthy": True, "event_count": 24, "market_count": 18, "token_count": 36},
                "clob": {"healthy": True, "rest_book_count": 18, "ws_connected": True},
            },
            "training_progress": {
                "tracked_examples": 320,
                "labeled_examples": 180,
                "pending_labels": 21,
                "final_resolution_labels": 12,
                "closed_trades": 24,
                "targets": {"labeled_examples": 250, "closed_trades": 50},
            },
            "model_league": {
                "leader_model_id": "hybrid_transition",
                "ranked_models": [
                    {
                        "model_id": "hybrid_transition",
                        "shadow_realized_pnl": 36.0,
                        "settled_count": 180,
                        "recent_learning_brier_lift": 0.014,
                        "strict_gate_pass": False,
                    }
                ],
            },
            "quote_freshness_sec": 6.5,
            "open_positions": [{"trade_id": "pmqf-1"}],
            "events": [],
            "closed_trades": [],
        }
    )
    store.write_readiness({"status": "paper_validating"})
    store.write_balance_history(
        [
            {"ts": "2026-03-06T00:00:00Z", "balance": 7500.0},
            {"ts": "2026-03-06T01:00:00Z", "balance": 7525.0},
        ]
    )
    store.write_models(
        [
            ModelShadowAccount(
                portfolio_id="polymarket_quantum_fold",
                model_id="hybrid_transition",
                shadow_starting_balance=7500.0,
                shadow_current_balance=7536.0,
                shadow_realized_pnl=36.0,
                shadow_roi_pct=0.48,
                settled_count=180,
                metrics={
                    "learning_tracked": 320,
                    "learning_settled": 180,
                    "strict_gate_pass": False,
                    "strict_gate_reason": "insufficient_labeled_examples",
                    "recent_learning_brier_lift": 0.014,
                    "rolling_200": {"brier_lift_abs": 0.014},
                },
                selected_for_execution=True,
            )
        ]
    )
    store.write_pid(2468)

    client = TestClient(command_center.app)
    portfolios = client.get("/api/portfolios").json()["portfolios"]
    summary = client.get("/api/portfolios/polymarket_quantum_fold/summary").json()
    state = client.get("/api/portfolios/polymarket_quantum_fold/state").json()

    assert any(item["portfolio_id"] == "polymarket_quantum_fold" for item in portfolios)
    assert summary["running"] is True
    assert summary["progress_pct"] > 0
    assert state["raw_state"]["source_health"]["gamma"]["healthy"] is True
    assert state["models"][0]["model_id"] == "hybrid_transition"


def test_command_center_betfair_strategy_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    betfair_store = PortfolioStateStore("betfair_core")
    betfair_store.write_account(
        build_strategy_account(
            portfolio_id="betfair_core",
            currency="EUR",
            starting_balance=1000.0,
            current_balance=1000.0,
            realized_pnl=0.0,
        )
    )
    betfair_store.write_state(
        {
            "portfolio_id": "betfair_core",
            "running": True,
            "status": "running",
            "mode": "paper",
            "strategy_books": {
                "betfair_suspension_lag": {
                    "strategy_id": "betfair_suspension_lag",
                    "label": "Suspension-Lag",
                    "candidate_count": 3,
                },
                "polymarket_binary_research": {
                    "strategy_id": "polymarket_binary_research",
                    "label": "Polymarket Binary Research",
                    "candidate_count": 5,
                }
            },
            "polymarket_signal_layer": {
                "feed_health": "healthy",
                "matched_events": 2,
            },
        }
    )
    betfair_store.write_readiness({"status": "monitoring"})

    client = TestClient(command_center.app)
    lag = client.get("/api/strategies/betfair_suspension_lag/state").json()
    poly_research = client.get("/api/strategies/polymarket_binary_research/state").json()
    poly = client.get("/api/signals/polymarket/state").json()

    assert lag["candidate_count"] == 3
    assert poly_research["candidate_count"] == 5
    assert poly["matched_events"] == 2


def test_command_center_rejects_start_for_monitor_only_strategy_views(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    client = TestClient(command_center.app)
    response = client.post("/api/portfolios/betfair_execution_book/start")

    assert response.status_code == 400
    payload = response.json()["detail"]
    assert payload["error"] == "monitor_only_portfolio"
    assert payload["parent_runner"] == "betfair_core"


def test_emit_snapshot_notifications_sends_closed_trade_alert(monkeypatch):
    sent = []
    monkeypatch.setattr(
        command_center,
        "_notifier",
        type(
            "Notifier",
            (),
            {
                "send_event": staticmethod(lambda **kwargs: sent.append(kwargs) or True),
                "send_digest": staticmethod(lambda *_args, **_kwargs: True),
            },
        )(),
    )
    command_center._snapshot_cache.clear()

    first = {
        "summary": {
            "portfolio_id": "cascade_alpha",
            "label": "Cascade Alpha",
            "running": True,
            "readiness": "paper_validating",
            "status": "running",
            "currency": "USD",
        },
        "state": {
            "recent_trades": [],
            "models": [],
        },
    }
    second = {
        "summary": dict(first["summary"]),
        "state": {
            "recent_trades": [
                {
                    "trade_id": "cascade-1",
                    "symbol": "TESTUSDT",
                    "side": "LONG",
                    "status": "CLOSED",
                    "net_pnl_usd": 12.5,
                    "close_reason": "take_profit",
                }
            ],
            "models": [],
        },
    }

    command_center._emit_snapshot_notifications([first])
    command_center._emit_snapshot_notifications([second])

    assert any(item["event_type"] == "trade_closed" for item in sent)


def test_emit_snapshot_notifications_sends_model_update_alert(monkeypatch):
    sent = []
    monkeypatch.setattr(
        command_center,
        "_notifier",
        type(
            "Notifier",
            (),
            {
                "send_event": staticmethod(lambda **kwargs: sent.append(kwargs) or True),
                "send_digest": staticmethod(lambda *_args, **_kwargs: True),
            },
        )(),
    )
    command_center._snapshot_cache.clear()

    first = {
        "summary": {
            "portfolio_id": "hedge_validation",
            "label": "Hedge Validation",
            "running": True,
            "readiness": "blocked",
            "status": "running",
            "currency": "USD",
        },
        "state": {
            "recent_trades": [],
            "models": [
                {
                    "model_id": "funding_online_learner",
                    "metrics": {
                        "last_retrain_time": "2026-03-06T10:00:00Z",
                        "last_retrain_result": "accepted",
                        "current_auc": 0.7,
                        "strict_gate_pass": False,
                        "rolling_200": {"brier_lift_abs": 0.01},
                        "settled_count": 10,
                    },
                }
            ],
        },
    }
    second = {
        "summary": dict(first["summary"]),
        "state": {
            "recent_trades": [],
            "models": [
                {
                    "model_id": "funding_online_learner",
                    "metrics": {
                        "last_retrain_time": "2026-03-06T11:00:00Z",
                        "last_retrain_result": "accepted",
                        "current_auc": 0.8,
                        "strict_gate_pass": True,
                        "rolling_200": {"brier_lift_abs": 0.05},
                        "settled_count": 20,
                    },
                }
            ],
        },
    }

    command_center._emit_snapshot_notifications([first])
    command_center._emit_snapshot_notifications([second])

    assert any(item["event_type"] == "model_update" for item in sent)


def test_command_center_history_trend(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    store = PortfolioStateStore("hedge_validation")
    path = store.runtime_dir / "summary_history.jsonl"
    store.append_jsonl(
        path,
        {
            "ts": "2026-03-05T12:00:00Z",
            "readiness": "blocked",
            "progress_pct": 20.0,
            "blocker_count": 4,
            "realized_pnl": 0.0,
            "roi_pct": 0.0,
            "open_count": 0,
        },
    )
    store.append_jsonl(
        path,
        {
            "ts": "2026-03-06T12:30:00Z",
            "readiness": "paper_validating",
            "progress_pct": 45.0,
            "blocker_count": 2,
            "realized_pnl": 10.0,
            "roi_pct": 0.02,
            "open_count": 1,
        },
    )
    store.append_jsonl(
        path,
        {
            "ts": "2026-03-06T18:30:00Z",
            "readiness": "paper_validating",
            "progress_pct": 55.0,
            "blocker_count": 1,
            "realized_pnl": 20.0,
            "roi_pct": 0.04,
            "open_count": 1,
        },
    )

    trend = command_center._history_trend("hedge_validation")

    assert trend["latest_progress_pct"] == 55.0
    assert trend["progress_delta_24h"] == 35.0
    assert trend["direction"] == "improving"
    assert trend["eta_hours"] is not None
    assert trend["eta_to_readiness"] != "unknown"


def test_command_center_enrich_models_adds_eta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    store = PortfolioStateStore("betfair_core")
    history_path = store.models_dir / "pure_logit_3" / "progress_history.jsonl"
    store.append_jsonl(
        history_path,
        {
            "ts": "2026-03-05T12:00:00Z",
            "settled_count": 20,
            "strict_gate_pass": False,
            "current_auc": 0.60,
            "brier_lift_abs": 0.01,
        },
    )
    store.append_jsonl(
        history_path,
        {
            "ts": "2026-03-06T12:00:00Z",
            "settled_count": 60,
            "strict_gate_pass": False,
            "current_auc": 0.64,
            "brier_lift_abs": 0.02,
        },
    )
    store.append_jsonl(
        history_path,
        {
            "ts": "2026-03-06T18:00:00Z",
            "settled_count": 85,
            "strict_gate_pass": False,
            "current_auc": 0.66,
            "brier_lift_abs": 0.03,
        },
    )

    models = command_center._enrich_models(
        "betfair_core",
        [
            {
                "portfolio_id": "betfair_core",
                "model_id": "pure_logit_3",
                "settled_count": 60,
                "metrics": {
                    "settled_count": 85,
                    "strict_gate_pass": False,
                    "strict_gate_reason": "insufficient_settled_bets",
                    "current_auc": 0.66,
                },
            }
        ],
        portfolio_eta_hours=18.0,
    )

    assert models[0]["settled_target"] == config.PREDICTION_STRICT_GATE_MIN_SETTLED
    assert models[0]["eta_to_readiness"] not in {"unknown", "quality_blocker"}
    assert models[0]["settled_remaining"] == config.PREDICTION_STRICT_GATE_MIN_SETTLED - 85


def test_trade_close_alert_filter_uses_threshold(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_MIN_TRADE_ALERT_PNL_USD", 5)
    summary = {"currency": "USD"}

    assert command_center._should_alert_trade_close({"net_pnl_usd": 6.0}, summary) is True
    assert command_center._should_alert_trade_close({"net_pnl_usd": 4.0}, summary) is False


def test_model_update_requires_gate_change_or_material_improvement(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_MODEL_ALERT_MIN_AUC_DELTA", 0.02)
    monkeypatch.setattr(config, "DISCORD_MODEL_ALERT_MIN_BRIER_LIFT_DELTA", 0.01)

    model = {
        "model_id": "m1",
        "metrics": {
            "last_retrain_time": "2026-03-06T11:00:00Z",
            "last_retrain_result": "accepted",
            "current_auc": 0.71,
            "strict_gate_pass": False,
            "rolling_200": {"brier_lift_abs": 0.011},
            "settled_count": 25,
        },
    }
    previous = {
        "last_retrain_time": "2026-03-06T10:00:00Z",
        "last_retrain_result": "accepted",
        "current_auc": 0.70,
        "strict_gate_pass": False,
        "rolling_200_brier_lift": 0.01,
        "settled_count": 20,
    }

    assert command_center._model_update_message(model, previous) is None

    model["metrics"]["current_auc"] = 0.74
    update = command_center._model_update_message(model, previous)

    assert update is not None
