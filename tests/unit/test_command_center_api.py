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
