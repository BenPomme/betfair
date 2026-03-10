from __future__ import annotations

from typing import Dict

import config
from factory.execution_bridge import FactoryExecutionBridge
from portfolio.types import PortfolioRunnerSpec


class _DummyManager:
    def __init__(self) -> None:
        self.running: Dict[str, int] = {}
        self.starts: list[str] = []

    def status(self, portfolio_id: str):
        pid = self.running.get(portfolio_id)
        return {"running": pid is not None, "pid": pid, "heartbeat": None}

    def start(self, portfolio_id: str):
        self.starts.append(portfolio_id)
        self.running[portfolio_id] = len(self.starts) + 1000
        return {"ok": True, "pid": self.running[portfolio_id]}


def test_execution_bridge_starts_enabled_targets_and_skips_disabled(monkeypatch):
    manager = _DummyManager()
    monkeypatch.setattr(config, "FACTORY_EXECUTION_AUTOSTART_ENABLED", True)
    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "full")

    specs = {
        "betfair_core": PortfolioRunnerSpec(
            portfolio_id="betfair_core",
            label="Betfair Core",
            category="betfair",
            control_mode="local_managed",
            currency="EUR",
            initial_balance=1000.0,
            enabled=True,
        ),
        "cascade_alpha": PortfolioRunnerSpec(
            portfolio_id="cascade_alpha",
            label="Cascade Alpha",
            category="crypto_alpha",
            control_mode="local_managed",
            currency="USD",
            initial_balance=1000.0,
            enabled=False,
        ),
    }

    def _fake_get_spec(portfolio_id: str):
        if portfolio_id not in specs:
            raise KeyError(portfolio_id)
        return specs[portfolio_id]

    monkeypatch.setattr("factory.execution_bridge.get_portfolio_spec", _fake_get_spec)

    bridge = FactoryExecutionBridge(process_manager=manager)
    state = {
        "lineages": [
            {
                "lineage_id": "betfair_prediction_value_league:challenger:1",
                "family_id": "betfair_prediction_value_league",
                "active": True,
                "current_stage": "paper",
                "role": "paper_challenger",
                "target_portfolios": ["betfair_core", "betfair_execution_book"],
            },
            {
                "lineage_id": "binance_cascade_regime:challenger:1",
                "family_id": "binance_cascade_regime",
                "active": True,
                "current_stage": "shadow",
                "role": "shadow_challenger",
                "target_portfolios": ["cascade_alpha"],
            },
        ]
    }

    payload = bridge.sync(state)

    assert payload["desired_portfolio_count"] == 2
    assert "betfair_core" in manager.starts
    statuses = {item["portfolio_id"]: item for item in payload["targets"]}
    assert statuses["betfair_core"]["status"] == "started"
    assert statuses["betfair_core"]["running"] is True
    assert statuses["betfair_core"]["requested_targets"] == ["betfair_core", "betfair_execution_book"]
    assert statuses["cascade_alpha"]["status"] == "runner_disabled"


def test_execution_bridge_respects_hard_stop(monkeypatch):
    manager = _DummyManager()
    monkeypatch.setattr(config, "FACTORY_EXECUTION_AUTOSTART_ENABLED", True)
    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "hard_stop")

    spec = PortfolioRunnerSpec(
        portfolio_id="betfair_core",
        label="Betfair Core",
        category="betfair",
        control_mode="local_managed",
        currency="EUR",
        initial_balance=1000.0,
        enabled=True,
    )
    monkeypatch.setattr("factory.execution_bridge.get_portfolio_spec", lambda portfolio_id: spec)

    bridge = FactoryExecutionBridge(process_manager=manager)
    payload = bridge.sync(
        {
            "lineages": [
                {
                    "lineage_id": "betfair_prediction_value_league:challenger:2",
                    "family_id": "betfair_prediction_value_league",
                    "active": True,
                    "current_stage": "paper",
                    "role": "paper_challenger",
                    "target_portfolios": ["betfair_core"],
                }
            ]
        }
    )

    assert manager.starts == []
    assert payload["targets"][0]["status"] == "factory_influence_paused"
