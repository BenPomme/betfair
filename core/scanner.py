"""
Overround computation and opportunity detection. Uses commission and stake_calculator;
rejects stale prices; emits Opportunity when net profit and liquidity pass thresholds.
"""
from decimal import Decimal
from typing import Optional, Callable, Any

import config
from core.types import PriceSnapshot, Opportunity
from core import commission as commission_module
from core import stake_calculator


def scan_snapshot(
    snapshot: PriceSnapshot,
    event_name: str = "",
    market_start: Any = None,
    min_net_profit_eur: Optional[Decimal] = None,
    min_liquidity_eur: Optional[Decimal] = None,
    max_stake_eur: Optional[Decimal] = None,
    pre_filter_threshold: Optional[Decimal] = None,
    mbr: Optional[Decimal] = None,
    discount_rate: Optional[Decimal] = None,
) -> Optional[Opportunity]:
    """
    Check a price snapshot for a back-back arb. If overround < threshold and
    profitable after commission and liquidity sufficient, return Opportunity.
    Uses config defaults for any None argument.
    """
    min_net_profit_eur = min_net_profit_eur or config.MIN_NET_PROFIT_EUR
    min_liquidity_eur = min_liquidity_eur or config.MIN_LIQUIDITY_EUR
    max_stake_eur = max_stake_eur or config.MAX_STAKE_EUR
    pre_filter_threshold = pre_filter_threshold or config.PRE_FILTER_THRESHOLD
    mbr = mbr or config.MBR
    discount_rate = discount_rate or config.DISCOUNT_RATE

    if not snapshot.selections or len(snapshot.selections) < 2:
        return None

    # Guard: skip markets with >3 selections (likely wrong market type or data error)
    if len(snapshot.selections) > 3:
        return None

    prices = [s.best_back_price for s in snapshot.selections]
    liquidities = [s.available_to_back for s in snapshot.selections]

    # Guard: all selections must have a back price (price > 0)
    if any(p <= Decimal("0") for p in prices):
        return None

    if any(l < min_liquidity_eur for l in liquidities):
        return None

    overround = sum(Decimal("1") / p for p in prices)
    if overround >= pre_filter_threshold:
        return None

    total_stake = max_stake_eur
    result = commission_module.evaluate_back_back_arb(
        prices, total_stake, mbr, discount_rate
    )
    if result is None:
        return None
    if result["min_net_profit"] < min_net_profit_eur:
        return None

    stakes = result["stakes"]

    # Check each leg's stake fits within available liquidity
    if any(stake > liq for stake, liq in zip(stakes, liquidities)):
        return None

    net_profits = result["net_profits"]
    # Use worst-case outcome for displayed gross/commission
    worst_idx = net_profits.index(min(net_profits))
    gross_profit = stakes[worst_idx] * prices[worst_idx] - sum(stakes)
    commission_eur = commission_module.commission(gross_profit, mbr, discount_rate)
    net_profit_eur = result["min_net_profit"]
    roi = result["roi"]

    selections_out = []
    for i, sel in enumerate(snapshot.selections):
        selections_out.append({
            "selection_id": sel.selection_id,
            "name": sel.name,
            "back_price": float(sel.best_back_price),
            "stake_eur": float(stakes[i]),
            "liquidity_eur": float(liquidities[i]),
        })

    actual_total = result["actual_total_stake"]
    return Opportunity(
        market_id=snapshot.market_id,
        event_name=event_name,
        market_start=market_start,
        arb_type="back_back",
        selections=tuple(selections_out),
        total_stake_eur=actual_total,
        overround_raw=overround,
        gross_profit_eur=gross_profit,
        commission_eur=commission_eur,
        net_profit_eur=net_profit_eur,
        net_roi_pct=roi,
        liquidity_by_selection=tuple(liquidities),
    )


def scan_snapshot_lay(
    snapshot: PriceSnapshot,
    event_name: str = "",
    market_start: Any = None,
    min_net_profit_eur: Optional[Decimal] = None,
    min_liquidity_eur: Optional[Decimal] = None,
    max_stake_eur: Optional[Decimal] = None,
    mbr: Optional[Decimal] = None,
    discount_rate: Optional[Decimal] = None,
) -> Optional[Opportunity]:
    """
    Check a price snapshot for a lay-lay arb. If lay_overround > 1.0 and
    profitable after commission and liquidity sufficient, return Opportunity.
    max_stake_eur is used as total lay stake budget (sum of all lay stakes).
    """
    min_net_profit_eur = min_net_profit_eur or config.MIN_NET_PROFIT_EUR
    min_liquidity_eur = min_liquidity_eur or config.MIN_LIQUIDITY_EUR
    max_stake_eur = max_stake_eur or config.MAX_STAKE_EUR
    mbr = mbr or config.MBR
    discount_rate = discount_rate or config.DISCOUNT_RATE

    if not snapshot.selections or len(snapshot.selections) < 2:
        return None

    # Guard: skip markets with >3 selections
    if len(snapshot.selections) > 3:
        return None

    lay_prices = [s.best_lay_price for s in snapshot.selections]
    lay_liquidities = [s.available_to_lay for s in snapshot.selections]

    # Skip if any lay price is 0 (no lay available)
    if any(lp <= Decimal("0") for lp in lay_prices):
        return None

    # Pre-filter: lay_overround must be > 1.0 for arb
    lay_overround = sum(Decimal("1") / lp for lp in lay_prices)
    if lay_overround <= Decimal("1"):
        return None

    # Check per-leg lay liquidity
    if any(liq < min_liquidity_eur for liq in lay_liquidities):
        return None

    total_liability = max_stake_eur
    result = commission_module.evaluate_lay_lay_arb(
        lay_prices, total_liability, mbr, discount_rate
    )
    if result is None:
        return None
    if result["min_net_profit"] < min_net_profit_eur:
        return None

    stakes = result["stakes"]

    # Check each leg's stake fits within available lay liquidity
    if any(stake > liq for stake, liq in zip(stakes, lay_liquidities)):
        return None

    net_profits = result["net_profits"]
    total_collected = result["total_collected"]
    # Use worst-case outcome for displayed gross/commission
    worst_idx = net_profits.index(min(net_profits))
    gross_profit = total_collected - stakes[worst_idx] * lay_prices[worst_idx]
    commission_eur = commission_module.commission(gross_profit, mbr, discount_rate)
    net_profit_eur = result["min_net_profit"]
    roi = result["roi"]

    selections_out = []
    for i, sel in enumerate(snapshot.selections):
        selections_out.append({
            "selection_id": sel.selection_id,
            "name": sel.name,
            "lay_price": float(sel.best_lay_price),
            "stake_eur": float(stakes[i]),
            "liability_eur": float(result["liabilities"][i]),
            "liquidity_eur": float(lay_liquidities[i]),
        })

    return Opportunity(
        market_id=snapshot.market_id,
        event_name=event_name,
        market_start=market_start,
        arb_type="lay_lay",
        selections=tuple(selections_out),
        total_stake_eur=total_collected,
        overround_raw=lay_overround,
        gross_profit_eur=gross_profit,
        commission_eur=commission_eur,
        net_profit_eur=net_profit_eur,
        net_roi_pct=roi,
        liquidity_by_selection=tuple(lay_liquidities),
    )


def scan_market(
    get_prices: Callable[[str], Optional[PriceSnapshot]],
    market_id: str,
    event_name: str = "",
    market_start: Any = None,
    **kwargs: Any,
) -> Optional[Opportunity]:
    """
    Get latest prices for market_id from cache; run both back-back and lay-lay
    scans. Return the opportunity with higher net profit, or None.
    """
    snapshot = get_prices(market_id)
    if snapshot is None:
        return None
    back_opp = scan_snapshot(snapshot, event_name=event_name, market_start=market_start, **kwargs)
    lay_opp = scan_snapshot_lay(snapshot, event_name=event_name, market_start=market_start, **kwargs)

    if back_opp and lay_opp:
        return back_opp if back_opp.net_profit_eur >= lay_opp.net_profit_eur else lay_opp
    return back_opp or lay_opp
