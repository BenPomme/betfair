"""
Build watchlist from market catalogue: LaLiga, Segunda Match Odds, 30-90 min before kick-off.
"""
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Any, Tuple


def get_watchlist_markets(
    catalogue_client: Any,
    event_type_ids: Optional[List[str]] = None,
    country_code: str = "ES",
    minutes_before_min: int = 30,
    minutes_before_max: int = 90,
    max_markets: int = 50,
) -> List[Tuple[str, str, Optional[Any]]]:
    """
    Return list of (market_id, event_name, market_start) for markets in the window
    30-90 min before start (configurable). Uses football (1) by default.
    catalogue_client: client that has list_events / list_market_catalogue (e.g. betfairlightweight APIClient).
    """
    from data.market_catalogue import get_market_catalogue
    markets = get_market_catalogue(
        client=catalogue_client,
        event_type_ids=event_type_ids or ["1"],  # Football
        country_code=country_code,
        max_results=max_markets,
    )
    now = datetime.now(timezone.utc)
    window_start = now + timedelta(minutes=minutes_before_min)
    window_end = now + timedelta(minutes=minutes_before_max)
    result = []
    for m in markets:
        start = m.get("market_start_time")
        if start is None:
            continue
        if hasattr(start, "replace"):
            start_naive = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        else:
            start_naive = start
        if not (window_start <= start_naive <= window_end):
            continue
        market_id = m.get("market_id")
        name = m.get("market_name") or m.get("event", {}).get("name", "") if isinstance(m.get("event"), dict) else ""
        if not name and m.get("event"):
            name = getattr(m["event"], "name", "") or ""
        result.append((str(market_id), name, start))
    return result


def get_watchlist_market_ids(
    catalogue_client: Any,
    **kwargs: Any,
) -> List[str]:
    """Return only market IDs for the watchlist."""
    return [m[0] for m in get_watchlist_markets(catalogue_client, **kwargs)]
