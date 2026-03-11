from __future__ import annotations

from decimal import Decimal

import config
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.types import PortfolioRunnerSpec


class _DummyRunner(PortfolioRunnerBase):
    def run(self) -> None:
        return None


def test_factory_config_overrides_preserve_config_types(monkeypatch):
    monkeypatch.setattr(config, "CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    monkeypatch.setattr(config, "CONTRARIAN_MIN_CONFIDENCE", 0.65)
    runner = _DummyRunner(
        PortfolioRunnerSpec(
            portfolio_id="dummy",
            label="Dummy",
            category="test",
            control_mode="disabled",
            currency="USD",
            initial_balance=1000.0,
        )
    )

    runner.apply_factory_config_overrides(
        {
            "CONTRARIAN_CAPITAL_PER_TRADE_PCT": 0.055,
            "CONTRARIAN_MIN_CONFIDENCE": "0.7",
        }
    )

    assert config.CONTRARIAN_CAPITAL_PER_TRADE_PCT == Decimal("0.055")
    assert isinstance(config.CONTRARIAN_CAPITAL_PER_TRADE_PCT, Decimal)
    assert config.CONTRARIAN_MIN_CONFIDENCE == 0.7
    assert isinstance(config.CONTRARIAN_MIN_CONFIDENCE, float)

    runner.restore_factory_config_overrides()

    assert config.CONTRARIAN_CAPITAL_PER_TRADE_PCT == Decimal("0.025")
    assert config.CONTRARIAN_MIN_CONFIDENCE == 0.65
