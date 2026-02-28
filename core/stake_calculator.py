"""
Equal-profit stake sizing for back-back arbitrage.
Uses commission module for net profit check; all Decimal, stakes rounded to 2 dp.
"""
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Tuple

from core import commission as commission_module


def compute_stakes(
    prices: List[Decimal],
    total_stake: Decimal,
    mbr: Decimal,
    discount: Decimal,
) -> Optional[Tuple[List[Decimal], List[Decimal], Decimal]]:
    """
    Compute equal-profit stakes for a back-back arb.
    Returns (stakes, net_profits, min_net_profit) if profitable after commission,
    else None. Stakes and amounts use 2 dp rounding.
    """
    result = commission_module.evaluate_back_back_arb(
        prices, total_stake, mbr, discount
    )
    if result is None:
        return None
    return (
        result["stakes"],
        result["net_profits"],
        result["min_net_profit"],
    )
