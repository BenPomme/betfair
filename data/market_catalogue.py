"""
REST client for Betfair market/event metadata. Can filter by sport/country or use whole exchange.
Includes discover_markets() for broad multi-sport scanning.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple

# Optional: use betfairlightweight when available
try:
    import betfairlightweight
    HAS_BETFAIR = True
except ImportError:
    HAS_BETFAIR = False

logger = logging.getLogger(__name__)

# Sport event type ID -> human name
SPORT_NAMES: Dict[str, str] = {
    "1": "Football",
    "2": "Tennis",
    "4": "Cricket",
    "7522": "Basketball",
    "7": "Horse Racing",
    "4339": "Greyhounds",
}


def get_market_catalogue(
    client: Optional[Any] = None,
    event_type_ids: Optional[List[str]] = None,
    country_code: Optional[str] = "ES",
    max_results: int = 50,
    all_sports: bool = False,
    market_type_codes: Optional[List[str]] = None,
    in_play_only: Optional[bool] = None,
) -> List[dict]:
    """
    Fetch market catalogue from Betfair REST API.
    client: betfairlightweight APIClient (must be logged in). If None, returns [].
    event_type_ids: e.g. ["1"] football, ["2"] tennis. If None and not all_sports, defaults to ["1","2"].
    country_code: e.g. ES. If None, no country filter.
    all_sports: if True, no event_type or country filter — whole exchange/sportsbook (subject to API limit).
    market_type_codes: e.g. ["MATCH_ODDS"] to filter market types.
    in_play_only: if True, only in-play markets; if False, only pre-match; if None, both.
    Returns list of dicts with market_id, market_name, event, runners, market_start_time.
    """
    if not HAS_BETFAIR or client is None:
        return []
    try:
        from betfairlightweight.filters import market_filter
        if all_sports:
            kwargs = {}
            if market_type_codes:
                kwargs["market_type_codes"] = market_type_codes
            if in_play_only is not None:
                kwargs["in_play_only"] = in_play_only
            market_filter_obj = market_filter(**kwargs)
        else:
            if event_type_ids is None:
                event_type_ids = ["1", "2"]
            kwargs = {"event_type_ids": event_type_ids}
            if country_code:
                kwargs["market_countries"] = [country_code]
            if market_type_codes:
                kwargs["market_type_codes"] = market_type_codes
            if in_play_only is not None:
                kwargs["in_play_only"] = in_play_only
            market_filter_obj = market_filter(**kwargs)
        result = client.betting.list_market_catalogue(
            filter=market_filter_obj,
            max_results=max_results,
            market_projection=["MARKET_DESCRIPTION", "RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
        )
        markets = []
        for m in result or []:
            mid = getattr(m, "market_id", None)
            if mid is None:
                continue
            markets.append({
                "market_id": mid,
                "market_name": getattr(m, "market_name", "") or "",
                "event": getattr(m, "event", None),
                "runners": getattr(m, "runners", []) or [],
                "market_start_time": getattr(m, "market_start_time", None),
                "_description": getattr(m, "description", None),
            })
        return markets
    except Exception as e:
        logger.warning("list_market_catalogue failed: %s", e)
        return []


def _event_name(market: dict) -> str:
    """Extract event name from a market catalogue dict."""
    ev = market.get("event")
    if ev is None:
        return ""
    if isinstance(ev, dict):
        return ev.get("name", "")
    return getattr(ev, "name", "") or ""


def discover_markets(
    client: Any,
    sport_configs: Optional[List[Dict[str, Any]]] = None,
    max_total: int = 500,
    include_in_play: bool = True,
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Discover all markets on the exchange for broad scanning.

    Fetches in batches of 200 (Betfair API limit with projections) using unfiltered
    queries. Extracts metadata from market description and event objects.
    The scanner's selection-count guard (2-3 only) filters out unsuitable markets.

    Args:
        client: betfairlightweight APIClient (logged in).
        sport_configs: ignored (kept for interface compat). Discovery is now unfiltered.
        max_total: maximum total markets to return.
        include_in_play: whether to also fetch in-play markets.

    Returns:
        (market_ids, market_metadata, runner_names) where:
        - market_metadata maps market_id -> {sport_name, country, event_name, ...}
        - runner_names maps market_id -> {selection_id -> runner_name}
    """
    BATCH_SIZE = 200  # Max per API call with projections before TOO_MUCH_DATA

    seen_ids: set = set()
    market_ids: List[str] = []
    metadata: Dict[str, Dict[str, str]] = {}
    runner_names: Dict[str, Dict[str, str]] = {}

    def _extract_and_add(markets: List[dict]) -> None:
        for m in markets:
            mid = str(m.get("market_id", ""))
            if not mid or mid in seen_ids:
                continue
            if len(market_ids) >= max_total:
                return
            seen_ids.add(mid)
            market_ids.append(mid)

            ev = m.get("event")
            country = ""
            event_name = ""
            event_id = ""
            if ev is not None:
                if isinstance(ev, dict):
                    country = ev.get("country_code", "")
                    event_name = ev.get("name", "")
                    event_id = str(ev.get("id", ""))
                else:
                    country = getattr(ev, "country_code", "") or ""
                    event_name = getattr(ev, "name", "") or ""
                    event_id = str(getattr(ev, "id", "") or "")

            desc = m.get("_description")
            market_type = ""
            if desc is not None:
                market_type = getattr(desc, "market_type", "") or ""

            mst = m.get("market_start_time")
            # Extract runner names for this market
            runners = m.get("runners", []) or []
            rn: Dict[str, str] = {}
            for r in runners:
                sid = str(getattr(r, "selection_id", "") or "")
                rname = getattr(r, "runner_name", "") or ""
                if sid and rname:
                    rn[sid] = rname
            if rn:
                runner_names[mid] = rn

            metadata[mid] = {
                "sport": "",
                "sport_name": market_type or "Unknown",
                "country": country,
                "event_id": event_id,
                "event_name": event_name,
                "market_name": m.get("market_name", ""),
                "market_start": mst.isoformat() if hasattr(mst, "isoformat") else str(mst) if mst else "",
                "market_type": market_type,
                "runner_count": len(m.get("runners", [])),
            }

    # Fetch 1: pre-match (all sports, all countries, no filters)
    try:
        pre_match_raw = get_market_catalogue(
            client=client,
            all_sports=True,
            max_results=min(max_total, BATCH_SIZE),
            in_play_only=False,
        )
        _extract_and_add(pre_match_raw)
        logger.info("discover_markets: %d pre-match markets", len(pre_match_raw))
    except Exception as e:
        logger.warning("discover_markets pre-match fetch failed: %s", e)

    # Fetch 2: in-play (separate call since in_play_only is a different filter)
    if include_in_play and len(market_ids) < max_total:
        try:
            in_play_raw = get_market_catalogue(
                client=client,
                all_sports=True,
                max_results=min(max_total - len(market_ids), BATCH_SIZE),
                in_play_only=True,
            )
            _extract_and_add(in_play_raw)
            logger.info("discover_markets: %d in-play markets", len(in_play_raw))
        except Exception as e:
            logger.warning("discover_markets in-play fetch failed: %s", e)

    logger.info("discover_markets: %d total markets, %d countries",
                len(market_ids), len(set(m.get("country", "") for m in metadata.values())))
    return market_ids, metadata, runner_names


def build_runner_name_map(raw_markets: List[dict]) -> Dict[str, Dict[str, str]]:
    """
    Build market_id -> {selection_id -> runner_name} mapping from catalogue data.
    Used to enrich PriceSnapshot with human-readable names since list_market_book
    does not return runner names.
    """
    name_map: Dict[str, Dict[str, str]] = {}
    for m in raw_markets:
        mid = str(m.get("market_id", ""))
        if not mid:
            continue
        runners = m.get("runners", []) or []
        runner_names: Dict[str, str] = {}
        for r in runners:
            sid = str(getattr(r, "selection_id", "") or "")
            rname = getattr(r, "runner_name", "") or ""
            if sid and rname:
                runner_names[sid] = rname
        if runner_names:
            name_map[mid] = runner_names
    return name_map


def get_all_event_type_ids(client: Optional[Any] = None) -> List[str]:
    """Return event type IDs (sports) that have markets. Use for whole-sportsbook scanning."""
    if not HAS_BETFAIR or client is None:
        return []
    try:
        from betfairlightweight.filters import market_filter
        result = client.betting.list_event_types(filter=market_filter())
        ids = []
        for r in result or []:
            et = getattr(r, "event_type", None)
            if et is not None and getattr(et, "id", None) is not None:
                ids.append(str(et.id))
        return ids
    except Exception as e:
        logger.warning("list_event_types failed: %s", e)
        return []


def get_events(
    client: Optional[Any] = None,
    event_type_ids: Optional[List[str]] = None,
    country_code: str = "ES",
) -> List[dict]:
    """
    Fetch events only (for building watchlist). Uses list_events.
    Returns list of dicts with event_id, name, open_date, etc.
    """
    if not HAS_BETFAIR or client is None:
        return []
    if event_type_ids is None:
        event_type_ids = ["1", "2"]
    try:
        from betfairlightweight.filters import event_filter
        event_filter_obj = event_filter(
            event_type_ids=event_type_ids,
            market_countries=[country_code],
        )
        result = client.betting.list_events(filter=event_filter_obj)
        events = []
        for e in result or []:
            events.append({
                "event_id": getattr(e, "event", {}).get("id"),
                "name": getattr(e, "event", {}).get("name", ""),
                "open_date": getattr(e, "event", {}).get("openDate"),
                "country_code": getattr(e, "event", {}).get("countryCode"),
            })
        return events
    except Exception:
        return []
