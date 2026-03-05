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
    include_experimental: bool = False,
) -> List[Tuple[str, str, str]]:
    """
    Find cross-market pairs within a single event.

    Args:
        event_markets: list of market_ids belonging to the same event.
        market_metadata: full metadata dict to look up market_type.

    Returns:
        List of (market_id_a, market_id_b, pair_type).

        Stable pair types:
        - mo_dnb

        Experimental pair types (only when include_experimental=True):
        - mo_ou25
        - mo_btts
        - cs_mo
    """
    mo_ids: List[str] = []
    dnb_ids: List[str] = []
    ou25_ids: List[str] = []
    btts_ids: List[str] = []
    cs_ids: List[str] = []

    for mid in event_markets:
        meta = market_metadata.get(mid, {})
        mt = meta.get("market_type", "")
        if mt == "MATCH_ODDS":
            mo_ids.append(mid)
        elif mt == "DRAW_NO_BET":
            dnb_ids.append(mid)
        elif mt == "OVER_UNDER_25":
            ou25_ids.append(mid)
        elif mt == "BOTH_TEAMS_TO_SCORE":
            btts_ids.append(mid)
        elif mt == "CORRECT_SCORE":
            cs_ids.append(mid)

    pairs: List[Tuple[str, str, str]] = []
    for mo_id in mo_ids:
        for dnb_id in dnb_ids:
            pairs.append((mo_id, dnb_id, "mo_dnb"))

    if include_experimental:
        for mo_id in mo_ids:
            for ou25_id in ou25_ids:
                pairs.append((mo_id, ou25_id, "mo_ou25"))
        for mo_id in mo_ids:
            for btts_id in btts_ids:
                pairs.append((mo_id, btts_id, "mo_btts"))
        for cs_id in cs_ids:
            for mo_id in mo_ids:
                pairs.append((cs_id, mo_id, "cs_mo"))
    return pairs
