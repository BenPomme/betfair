#!/usr/bin/env python3
"""
Test the pipeline with real Betfair data (REST API).
Uses your Delayed Application Key — login, fetch markets & prices, run scanner.
Run from project root: python scripts/test_real_data.py

Login uses locale=spain (identitysso.betfair.es) by default. If login fails:
- INVALID_USERNAME_OR_PASSWORD: ensure credentials are for betfair.es; special chars in
  username/password must be URL-encoded per Betfair docs.
- CERT_AUTH_REQUIRED: use certificate login — create a self-signed cert, upload the .crt
  to your Betfair account, set BF_CERTS_PATH to the folder with your .crt and .key.
"""
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional

import config
from core.types import PriceSnapshot, SelectionPrice
from core.scanner import scan_snapshot
from data.price_cache import PriceCache
from data.market_catalogue import get_market_catalogue


def _market_book_to_snapshot(market_id: str, market_book) -> Optional[PriceSnapshot]:
    """Convert betfairlightweight market_book to PriceSnapshot."""
    if not market_book or not getattr(market_book, "runners", None):
        return None
    selections = []
    for runner in market_book.runners:
        ex = getattr(runner, "ex", None)
        if not ex or not getattr(ex, "available_to_back", None):
            continue
        atb = ex.available_to_back
        if not atb:
            continue
        best_price = Decimal(str(atb[0].price))
        available = Decimal(str(atb[0].size))
        selection_id = str(getattr(runner, "selection_id", ""))
        name = getattr(runner, "runner_name", "") or selection_id
        selections.append(SelectionPrice(
            selection_id=selection_id,
            name=name,
            best_back_price=best_price,
            available_to_back=available,
        ))
    if not selections:
        return None
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


def main():
    try:
        from data.betfair_client import create_and_login
        from betfairlightweight.filters import price_projection, price_data
    except ImportError as e:
        print("Missing dependency:", e)
        sys.exit(1)

    print("Logging in to Betfair...")
    try:
        trading = create_and_login()
    except Exception as e:
        print("Login failed:", e)
        sys.exit(1)

    print("Fetching market catalogue (football, ES)...")
    markets = get_market_catalogue(
        client=trading,
        event_type_ids=["1"],
        country_code="ES",
        max_results=10,
    )
    if not markets:
        print("No markets returned. Check filters or try different event type.")
        trading.logout()
        sys.exit(0)

    market_ids = [m["market_id"] for m in markets[:5]]
    print(f"Fetching prices for {len(market_ids)} markets: {market_ids}")

    try:
        market_books = trading.betting.list_market_book(
            market_ids=market_ids,
            price_projection=price_projection(price_data=price_data(ex_all_offers=True)),
        )
    except Exception as e:
        print("list_market_book failed:", e)
        trading.logout()
        sys.exit(1)

    cache = PriceCache(max_age_seconds=300)  # 5 min for this test
    books_list = market_books if isinstance(market_books, list) else getattr(market_books, "result", []) or []
    for book in books_list:
        mid = str(getattr(book, "market_id", ""))
        snap = _market_book_to_snapshot(mid, book)
        if snap:
            cache.set_prices(snap)
            overround = sum(Decimal("1") / s.best_back_price for s in snap.selections)
            opp = scan_snapshot(snap)
            if opp:
                print(f"  {mid}  overround={overround:.4f}  OPPORTUNITY  net_profit_eur={opp.net_profit_eur}")
            else:
                print(f"  {mid}  overround={overround:.4f}  no arb")

    trading.logout()
    print("Done. Pipeline ran on real data.")


if __name__ == "__main__":
    main()
