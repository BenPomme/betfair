#!/usr/bin/env python3
"""
Fetch at least one market ID using the same resolution chain as the engine
(watchlist → catalogue ES → catalogue any → all sports → event types),
then append MARKET_IDS=id1,id2,... to .env or print the line for the user.
Run from project root, or the script sets cwd and loads .env automatically.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass


def main() -> None:
    from data.betfair_client import create_and_login
    from data.market_catalogue import get_market_catalogue, get_all_event_type_ids
    from strategy.market_selector import get_watchlist_market_ids

    client = create_and_login()
    try:
        market_ids_str = os.getenv("MARKET_IDS", "")
        market_ids = [m.strip() for m in market_ids_str.split(",") if m.strip()]

        if not market_ids:
            try:
                market_ids = get_watchlist_market_ids(
                    client,
                    event_type_ids=["1"],
                    country_code="ES",
                    minutes_before_min=0,
                    minutes_before_max=120,
                    max_markets=30,
                )
            except Exception:
                pass
        if not market_ids:
            try:
                catalogue = get_market_catalogue(
                    client=client,
                    event_type_ids=["1", "2"],
                    country_code="ES",
                    max_results=30,
                )
                market_ids = [str(m.get("market_id")) for m in catalogue if m.get("market_id")]
            except Exception:
                pass
        if not market_ids:
            try:
                catalogue = get_market_catalogue(
                    client=client,
                    event_type_ids=["1", "2"],
                    country_code=None,
                    max_results=30,
                )
                market_ids = [str(m.get("market_id")) for m in catalogue if m.get("market_id")]
            except Exception:
                pass
        if not market_ids:
            try:
                catalogue = get_market_catalogue(
                    client=client,
                    all_sports=True,
                    max_results=100,
                )
                market_ids = [str(m.get("market_id")) for m in catalogue if m.get("market_id")]
            except Exception:
                pass
        if not market_ids:
            event_type_ids = get_all_event_type_ids(client)
            for et_id in (event_type_ids or [])[:5]:
                if market_ids:
                    break
                try:
                    catalogue = get_market_catalogue(
                        client=client,
                        event_type_ids=[et_id],
                        country_code=None,
                        max_results=20,
                    )
                    market_ids = [str(m.get("market_id")) for m in catalogue if m.get("market_id")]
                except Exception:
                    pass

        if not market_ids:
            print("No markets found. Check credentials and API; run list_markets.py to inspect.")
            return

        line = "MARKET_IDS=" + ",".join(market_ids)
        env_path = project_root / ".env"
        if env_path.exists():
            content = env_path.read_text()
            if "MARKET_IDS=" in content:
                lines = []
                for raw in content.splitlines():
                    if raw.strip().startswith("MARKET_IDS="):
                        lines.append(line)
                    else:
                        lines.append(raw)
                env_path.write_text("\n".join(lines) + "\n")
                print("Updated MARKET_IDS in .env")
            else:
                with open(env_path, "a") as f:
                    f.write("\n" + line + "\n")
                print("Appended MARKET_IDS to .env")
        else:
            print("No .env file; add this line to your environment or .env:")
            print(line)
        print("Market IDs:", ", ".join(market_ids[:10]), "..." if len(market_ids) > 10 else "")
    finally:
        try:
            client.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
