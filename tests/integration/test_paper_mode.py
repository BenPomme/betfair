"""
Integration test: synthetic prices (overround < threshold) -> scanner -> executor (paper).
Assert log entry exists and commission/stakes match commission module.
"""
from datetime import datetime, timezone
from decimal import Decimal
import os

import pytest

# Force paper mode for this test
os.environ["PAPER_TRADING"] = "true"

from core.types import PriceSnapshot, SelectionPrice
from core.scanner import scan_market
from core.commission import commission, MBR
from execution.executor import execute_opportunity
from execution.paper_executor import PaperExecutor
from data.price_cache import PriceCache


def test_paper_mode_integration():
    # 1. Build synthetic snapshot with overround < 0.97
    prices = [Decimal("3.20"), Decimal("3.20"), Decimal("3.20")]
    liquidity = Decimal("1000")
    selections = [
        SelectionPrice(str(i), f"Sel{i}", p, liquidity)
        for i, p in enumerate(prices)
    ]
    snapshot = PriceSnapshot(
        market_id="1.integration",
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )

    # 2. Cache and scanner
    cache = PriceCache(max_age_seconds=10)
    cache.set_prices(snapshot)

    opp = scan_market(cache.get_prices, "1.integration", event_name="Integration Event")
    assert opp is not None

    # 3. Execute (paper)
    paper_executor = PaperExecutor()
    result = execute_opportunity(opp, paper_executor=paper_executor)
    assert result is not None

    # 4. Assert log entry and commission match
    assert result["mode"] == "paper"
    assert result["market_id"] == "1.integration"
    assert result["net_profit_eur"] > 0
    expected_commission = commission(opp.gross_profit_eur, MBR, Decimal("0"))
    assert abs(result["commission_eur"] - float(expected_commission)) < 0.01
    assert len(paper_executor.log_entries) == 1
    assert paper_executor.balance == opp.net_profit_eur
