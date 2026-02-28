#!/usr/bin/env python3
"""
Print market IDs from Betfair so you can set MARKET_IDS in .env if the dashboard
says "No markets to watch". Run: python scripts/list_markets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.betfair_client import create_and_login
from data.market_catalogue import get_market_catalogue


def main():
    print("Logging in...")
    try:
        client = create_and_login()
    except Exception as e:
        print("Login failed:", e)
        sys.exit(1)

    for label, kwargs in [
        ("Football+Tennis (ES)", {"event_type_ids": ["1", "2"], "country_code": "ES", "max_results": 30}),
        ("Football+Tennis (any country)", {"event_type_ids": ["1", "2"], "country_code": None, "max_results": 30}),
        ("Whole exchange (all sports)", {"all_sports": True, "max_results": 100}),
    ]:
        cat = get_market_catalogue(client=client, **kwargs)
        print(f"\n{label}: {len(cat)} markets")
        for m in cat[:12]:
            mid = m.get("market_id")
            name = (m.get("market_name") or "")[:55]
            print(f"  {mid}  {name}")
        if len(cat) > 12:
            print(f"  ... and {len(cat) - 12} more")

    client.logout()
    print("\nTo use specific markets, add to .env:")
    print("  MARKET_IDS=1.123456,1.234567")


if __name__ == "__main__":
    main()
