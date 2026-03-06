from __future__ import annotations

from portfolio.state_store import PortfolioStateStore
from portfolio.types import ModelShadowAccount
from portfolio.accounting import build_strategy_account
import config


def test_portfolio_state_store_keeps_accounts_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(tmp_path))

    betfair_store = PortfolioStateStore("betfair_core")
    hedge_store = PortfolioStateStore("hedge_validation")

    betfair_account = build_strategy_account(
        portfolio_id="betfair_core",
        currency="EUR",
        starting_balance=1000.0,
        current_balance=1012.5,
        realized_pnl=12.5,
    )
    hedge_account = build_strategy_account(
        portfolio_id="hedge_validation",
        currency="USD",
        starting_balance=50000.0,
        current_balance=50040.0,
        realized_pnl=40.0,
    )

    betfair_store.write_account(betfair_account)
    hedge_store.write_account(hedge_account)
    betfair_store.write_models([
        ModelShadowAccount(
            portfolio_id="betfair_core",
            model_id="residual_logit",
            shadow_starting_balance=100000.0,
            shadow_current_balance=100250.0,
            shadow_realized_pnl=250.0,
            shadow_roi_pct=0.25,
            settled_count=50,
        )
    ])

    assert betfair_store.read_account().currency == "EUR"
    assert hedge_store.read_account().currency == "USD"
    assert hedge_store.read_models() == []
    assert betfair_store.read_models()[0]["model_id"] == "residual_logit"
