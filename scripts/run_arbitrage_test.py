#!/usr/bin/env python3
"""
Run the full arbitrage pipeline once with synthetic data: no Betfair login needed.
Shows scanner → risk manager → paper executor. Use this to test that trading works.
Run from project root: python scripts/run_arbitrage_test.py
"""
import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.types import PriceSnapshot, SelectionPrice
from data.price_cache import PriceCache
from core.scanner import scan_market
from core.risk_manager import RiskManager
from execution.executor import execute_opportunity
from execution.paper_executor import PaperExecutor


def make_snapshot(market_id: str, prices: list, liquidity_eur: Decimal = Decimal("1000")):
    """Build a PriceSnapshot for testing."""
    selections = [
        SelectionPrice(str(i), f"Selection_{i}", Decimal(str(p)), liquidity_eur)
        for i, p in enumerate(prices)
    ]
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


def main():
    print("=== Arbitrage trading test (synthetic data) ===\n")
    print("PAPER_TRADING =", config.PAPER_TRADING)
    if not config.PAPER_TRADING:
        print("Set PAPER_TRADING=true in .env to run.")
        sys.exit(1)

    # 1) Cache with two "markets": one with arb, one without
    cache = PriceCache(max_age_seconds=60)
    # Market A: overround 0.9375 → arb
    snap_arb = make_snapshot("1.arb-test", [3.2, 3.2, 3.2], liquidity_eur=Decimal("500"))
    cache.set_prices(snap_arb)
    # Market B: overround > 1 → no arb
    snap_no = make_snapshot("1.no-arb-test", [1.9, 2.0, 2.1], liquidity_eur=Decimal("500"))
    cache.set_prices(snap_no)

    risk_manager = RiskManager(
        max_stake_eur=config.MAX_STAKE_EUR,
        daily_loss_limit_eur=config.DAILY_LOSS_LIMIT_EUR,
    )
    paper_executor = PaperExecutor(initial_balance_eur=Decimal("1000"))

    market_ids = ["1.arb-test", "1.no-arb-test"]
    executed = 0

    for market_id in market_ids:
        opp = scan_market(cache.get_prices, market_id, event_name=f"Test {market_id}")
        if opp is None:
            print(f"  {market_id}: no opportunity (overround too high or stale)")
            continue
        if not risk_manager.can_execute(opp):
            print(f"  {market_id}: opportunity found but risk manager blocked")
            continue
        result = execute_opportunity(opp, paper_executor=paper_executor)
        if result:
            executed += 1
            risk_manager.register_execution(opp, opp.net_profit_eur)
            print(f"  {market_id}: ARB EXECUTED (paper)")
            print(f"    net_profit_eur = {result['net_profit_eur']}")
            print(f"    total_stake_eur = {result['total_stake_eur']}")
            print(f"    overround_raw = {result['overround_raw']}")

    print()
    print("Paper executor balance:", paper_executor.balance)
    print("Trades executed:", executed)
    print("Log entries:", len(paper_executor.log_entries))
    if paper_executor.log_entries:
        print("\nLast paper log (summary):")
        last = paper_executor.log_entries[-1]
        for k in ["market_id", "total_stake_eur", "net_profit_eur", "commission_eur", "fill_simulated_optimistic"]:
            if k in last:
                print(f"  {k}: {last[k]}")
    print("\n=== Arbitrage test done ===")


if __name__ == "__main__":
    main()
