"""
Filter out markets that don't meet minimum liquidity at best back price.
"""
from decimal import Decimal
from typing import Callable, List, Optional

from core.types import PriceSnapshot


def filter_by_liquidity(
    market_ids: List[str],
    get_prices: Callable[[str], Optional[PriceSnapshot]],
    min_liquidity_eur: Decimal,
) -> List[str]:
    """
    Return market_ids for which get_prices(mid) returns a snapshot and every
    selection has available_to_back >= min_liquidity_eur.
    """
    result = []
    for mid in market_ids:
        snap = get_prices(mid)
        if snap is None:
            continue
        if all(s.available_to_back >= min_liquidity_eur for s in snap.selections):
            result.append(mid)
    return result
