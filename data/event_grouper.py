"""
Group markets by event_id and find cross-market pairs for arbitrage scanning.
"""
from typing import Dict, List, Tuple


def group_by_event(market_metadata: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Group market IDs by their event_id.

    Args:
        market_metadata: mapping of market_id -> metadata dict (must contain "event_id").

    Returns:
        Dict mapping event_id -> list of market_ids sharing that event.
        Skips markets with empty/missing event_id.
    """
    groups: Dict[str, List[str]] = {}
    for market_id, meta in market_metadata.items():
        event_id = meta.get("event_id", "")
        if not event_id:
            continue
        groups.setdefault(event_id, []).append(market_id)
    return groups


def get_cross_market_pairs(
    event_markets: List[str],
    market_metadata: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str]]:
    """
    Find (MATCH_ODDS_market_id, DRAW_NO_BET_market_id) pairs within a single event.

    Args:
        event_markets: list of market_ids belonging to the same event.
        market_metadata: full metadata dict to look up market_type.

    Returns:
        List of (mo_market_id, dnb_market_id) tuples. May return multiple pairs
        if the event has multiple MATCH_ODDS or DNB markets.
    """
    mo_ids: List[str] = []
    dnb_ids: List[str] = []

    for mid in event_markets:
        meta = market_metadata.get(mid, {})
        mt = meta.get("market_type", "")
        if mt == "MATCH_ODDS":
            mo_ids.append(mid)
        elif mt == "DRAW_NO_BET":
            dnb_ids.append(mid)

    pairs: List[Tuple[str, str]] = []
    for mo_id in mo_ids:
        for dnb_id in dnb_ids:
            pairs.append((mo_id, dnb_id))
    return pairs
