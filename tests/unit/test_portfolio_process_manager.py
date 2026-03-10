from __future__ import annotations

import config
from monitoring.portfolio_process_manager import PortfolioProcessManager


def test_process_manager_blocks_research_factory_start_in_hard_stop(monkeypatch):
    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "hard_stop")

    manager = PortfolioProcessManager()
    result = manager.start("research_factory")

    assert result == {"ok": False, "error": "agentic_factory_hard_stopped"}


def test_process_manager_returns_unknown_portfolio_error_for_unmanaged_ids():
    manager = PortfolioProcessManager()

    assert manager.start("betfair_execution_book") == {
        "ok": False,
        "error": "unknown_portfolio",
        "portfolio_id": "betfair_execution_book",
    }
    assert manager.stop("betfair_execution_book") == {
        "ok": False,
        "error": "unknown_portfolio",
        "portfolio_id": "betfair_execution_book",
    }
    assert manager.restart("betfair_execution_book") == {
        "ok": False,
        "error": "unknown_portfolio",
        "portfolio_id": "betfair_execution_book",
    }
