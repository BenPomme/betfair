"""Unit tests for execution.paper_executor. Virtual ledger, log schema, commission match."""
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from core.types import Opportunity
from execution.paper_executor import PaperExecutor, _log_entry
from core.commission import commission, MBR


def _make_opportunity():
    return Opportunity(
        market_id="1.123",
        event_name="Test Event",
        market_start=datetime.now(timezone.utc),
        arb_type="back_back",
        selections=(
            {"name": "A", "back_price": 3.2, "stake_eur": 33.33, "liquidity_eur": 500},
            {"name": "B", "back_price": 3.2, "stake_eur": 33.33, "liquidity_eur": 500},
            {"name": "C", "back_price": 3.2, "stake_eur": 33.33, "liquidity_eur": 500},
        ),
        total_stake_eur=Decimal("99.99"),
        overround_raw=Decimal("0.9375"),
        gross_profit_eur=Decimal("6.66"),
        commission_eur=Decimal("0.33"),
        net_profit_eur=Decimal("6.33"),
        net_roi_pct=Decimal("0.0633"),
        liquidity_by_selection=(Decimal("500"), Decimal("500"), Decimal("500")),
    )


def test_paper_executor_log_schema_fields():
    opp = _make_opportunity()
    exec = PaperExecutor()
    entry = exec.log(opp)
    assert "ts" in entry
    assert entry["mode"] == "paper"
    assert entry["market_id"] == opp.market_id
    assert entry["event"] == opp.event_name
    assert entry["selections"] is not None
    assert entry["total_stake_eur"] == float(opp.total_stake_eur)
    assert entry["overround_raw"] == float(opp.overround_raw)
    assert entry["gross_profit_eur"] == float(opp.gross_profit_eur)
    assert entry["commission_eur"] == float(opp.commission_eur)
    assert entry["net_profit_eur"] == float(opp.net_profit_eur)
    assert "fill_simulated_optimistic" in entry
    assert "fill_simulated_realistic_net" in entry


def test_paper_executor_commission_matches_module():
    opp = _make_opportunity()
    expected_commission = commission(opp.gross_profit_eur, MBR, Decimal("0"))
    entry = _log_entry(opp)
    assert abs(entry["commission_eur"] - float(expected_commission)) < 0.01


def test_paper_executor_virtual_ledger():
    opp = _make_opportunity()
    exec = PaperExecutor(initial_balance_eur=Decimal("1000"))
    exec.log(opp)
    # Balance: 1000 - total_stake + total_stake + net_profit = 1000 + net_profit
    assert exec.balance == Decimal("1000") + opp.net_profit_eur
    assert len(exec.log_entries) == 1
    exec.log(opp)
    assert len(exec.log_entries) == 2
    assert exec.balance == Decimal("1000") + opp.net_profit_eur + opp.net_profit_eur
