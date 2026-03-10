from __future__ import annotations

from fastapi.testclient import TestClient

import config
from monitoring import command_center
from portfolio.accounting import build_strategy_account
from portfolio.state_store import PortfolioStateStore


class _DummyManager:
    def status(self, portfolio_id: str):
        store = PortfolioStateStore(portfolio_id)
        return {"running": store.read_pid() is not None, "pid": store.read_pid(), "heartbeat": store.read_heartbeat()}


def test_command_center_exposes_research_factory_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    store = PortfolioStateStore("research_factory")
    store.write_account(
        build_strategy_account(
            portfolio_id="research_factory",
            currency="USD",
            starting_balance=100000.0,
            current_balance=100250.0,
            realized_pnl=250.0,
            trade_count=5,
        )
    )
    store.write_state(
        {
            "portfolio_id": "research_factory",
            "running": True,
            "status": "running",
            "mode": "research",
            "research_summary": {
                "family_count": 5,
                "lineage_count": 5,
                "paper_pnl": 250.0,
            },
            "manifests": {"pending": [], "live_loadable": []},
            "queue": [],
        }
    )
    store.write_readiness({"status": "paper_validating", "blockers": ["human_signoff_required_for_live"]})
    store.write_balance_history(
        [
            {"ts": "2026-03-06T00:00:00Z", "balance": 100000.0},
            {"ts": "2026-03-06T01:00:00Z", "balance": 100250.0},
        ]
    )
    store.write_pid(7777)

    client = TestClient(command_center.app)
    portfolios = client.get("/api/portfolios").json()["portfolios"]
    summary = client.get("/api/portfolios/research_factory/summary").json()

    assert any(item["portfolio_id"] == "research_factory" for item in portfolios)
    assert summary["portfolio_id"] == "research_factory"
    assert summary["running"] is True
    assert summary["readiness"] == "paper_validating"


def test_command_center_marks_research_factory_as_intentionally_paused_in_hard_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "hard_stop")
    monkeypatch.setattr(command_center, "_process_manager", _DummyManager())

    client = TestClient(command_center.app)
    summary = client.get("/api/portfolios/research_factory/summary").json()
    state = client.get("/api/portfolios/research_factory/state").json()

    assert summary["portfolio_id"] == "research_factory"
    assert summary["running"] is False
    assert summary["status"] == "paused"
    assert summary["readiness"] == "research_only"
    assert state["raw_state"]["agentic_factory_mode"] == "hard_stop"
    assert state["raw_state"]["factory_influence_allowed"] is False
    assert "agentic_factory_hard_stopped" in state["readiness"]["blockers"]


def test_command_center_exposes_git_branch_divergence(monkeypatch):
    monkeypatch.setattr(command_center, "_git_sha", lambda: "abc1234")
    monkeypatch.setattr(command_center, "_git_branch", lambda: "codex/agentic-factory-experiments")
    monkeypatch.setattr(
        command_center,
        "_git_main_divergence",
        lambda: {
            "branch": "codex/agentic-factory-experiments",
            "ahead_of_origin_main": 1,
            "behind_origin_main": 7,
            "origin_main_sha": "deadbee",
        },
    )

    client = TestClient(command_center.app)
    payload = client.get("/api/portfolios").json()

    assert payload["git_sha"] == "abc1234"
    assert payload["git_branch"] == "codex/agentic-factory-experiments"
    assert payload["git_main_divergence"]["ahead_of_origin_main"] == 1
    assert payload["git_main_divergence"]["behind_origin_main"] == 7
