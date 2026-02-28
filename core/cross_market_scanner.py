"""
Cross-market arbitrage scanner: finds pricing inconsistencies between
related markets on the SAME event (e.g. MATCH_ODDS vs DRAW_NO_BET).

Tier 3 — financially critical. All arithmetic uses Decimal.
"""
from decimal import Decimal
from typing import Any, Optional

import config
from core.types import PriceSnapshot, Opportunity
from core import commission as commission_module


def _match_selections_by_name(
    mo_snapshot: PriceSnapshot,
    dnb_snapshot: PriceSnapshot,
) -> tuple:
    """
    Match selections between MATCH_ODDS and DRAW_NO_BET by name.

    Returns (pairs, draw_sel) where:
      - pairs: list of (mo_selection, dnb_selection) for Home and Away
      - draw_sel: the Draw SelectionPrice from MO, or None if not found
    """
    # Build name -> selection lookup for DNB
    dnb_by_name = {}
    for sel in dnb_snapshot.selections:
        key = sel.name.strip().lower()
        dnb_by_name[key] = sel

    pairs = []
    draw_sel = None
    for mo_sel in mo_snapshot.selections:
        key = mo_sel.name.strip().lower()
        if key == "the draw" or key == "draw":
            draw_sel = mo_sel
            continue
        dnb_sel = dnb_by_name.get(key)
        if dnb_sel is not None:
            pairs.append((mo_sel, dnb_sel))

    return pairs, draw_sel


def scan_cross_market(
    mo_snapshot: PriceSnapshot,
    dnb_snapshot: PriceSnapshot,
    event_name: str = "",
    market_start: Any = None,
    min_net_profit_eur: Optional[Decimal] = None,
    min_liquidity_eur: Optional[Decimal] = None,
    max_stake_eur: Optional[Decimal] = None,
    mbr: Optional[Decimal] = None,
    discount_rate: Optional[Decimal] = None,
) -> Optional[Opportunity]:
    """
    Scan for cross-market arb between MATCH_ODDS and DRAW_NO_BET markets.

    For each matched selection (Home, Away), checks both directions:
      1. Back in MO, Lay in DNB
      2. Back in DNB, Lay in MO

    Uses evaluate_back_lay_arb() which handles double commission (both Betfair).
    Returns the best profitable Opportunity or None.
    """
    min_net_profit_eur = min_net_profit_eur or config.MIN_NET_PROFIT_EUR
    min_liquidity_eur = min_liquidity_eur or config.MIN_LIQUIDITY_EUR
    max_stake_eur = max_stake_eur or config.MAX_STAKE_EUR
    mbr = mbr or config.MBR
    discount_rate = discount_rate if discount_rate is not None else config.DISCOUNT_RATE

    # Validate snapshots
    if not mo_snapshot or not dnb_snapshot:
        return None
    if not mo_snapshot.selections or not dnb_snapshot.selections:
        return None

    # MO should have 3 selections, DNB should have 2
    if len(mo_snapshot.selections) != 3 or len(dnb_snapshot.selections) != 2:
        return None

    pairs, draw_sel = _match_selections_by_name(mo_snapshot, dnb_snapshot)
    if len(pairs) < 2:
        return None  # Need both Home and Away matched

    best_opp: Optional[Opportunity] = None
    best_net: Decimal = Decimal("0")

    for mo_sel, dnb_sel in pairs:
        # Direction 1: Back in MO, Lay in DNB — requires 3-leg with draw hedge
        # Only attempt if draw selection has a valid back price and sufficient liquidity
        if (
            draw_sel is not None
            and mo_sel.best_back_price > Decimal("0")
            and dnb_sel.best_lay_price > Decimal("0")
            and draw_sel.best_back_price > Decimal("0")
        ):
            if (
                mo_sel.available_to_back >= min_liquidity_eur
                and dnb_sel.available_to_lay >= min_liquidity_eur
                and draw_sel.available_to_back >= min_liquidity_eur
            ):
                result = commission_module.evaluate_mo_dnb_3leg_arb(
                    mo_back_price=mo_sel.best_back_price,
                    dnb_lay_price=dnb_sel.best_lay_price,
                    draw_back_price=draw_sel.best_back_price,
                    max_stake=max_stake_eur,
                    mbr=mbr,
                    discount=discount_rate,
                )
                if result is not None and result["net_profit"] >= min_net_profit_eur:
                    if result["net_profit"] > best_net:
                        best_net = result["net_profit"]
                        best_opp = _build_opportunity_3leg(
                            mo_market_id=mo_snapshot.market_id,
                            dnb_market_id=dnb_snapshot.market_id,
                            event_name=event_name,
                            market_start=market_start,
                            selection_name=mo_sel.name,
                            mo_back_price=mo_sel.best_back_price,
                            dnb_lay_price=dnb_sel.best_lay_price,
                            draw_back_price=draw_sel.best_back_price,
                            back_liquidity=mo_sel.available_to_back,
                            lay_liquidity=dnb_sel.available_to_lay,
                            draw_liquidity=draw_sel.available_to_back,
                            result=result,
                        )

        # Direction 2: Back in DNB, Lay in MO
        # Draw is the BEST scenario here (DNB voided, MO lay wins) so the
        # 2-outcome model already captures the two worst cases — no draw hedge needed.
        if dnb_sel.best_back_price > Decimal("0") and mo_sel.best_lay_price > Decimal("0"):
            if dnb_sel.available_to_back >= min_liquidity_eur and mo_sel.available_to_lay >= min_liquidity_eur:
                result = commission_module.evaluate_back_lay_arb(
                    back_price=dnb_sel.best_back_price,
                    lay_price=mo_sel.best_lay_price,
                    max_stake=max_stake_eur,
                    mbr=mbr,
                    discount=discount_rate,
                )
                if result is not None and result["net_profit"] >= min_net_profit_eur:
                    if result["net_profit"] > best_net:
                        best_net = result["net_profit"]
                        best_opp = _build_opportunity(
                            mo_market_id=mo_snapshot.market_id,
                            dnb_market_id=dnb_snapshot.market_id,
                            event_name=event_name,
                            market_start=market_start,
                            selection_name=dnb_sel.name,
                            direction="back_dnb_lay_mo",
                            back_price=dnb_sel.best_back_price,
                            lay_price=mo_sel.best_lay_price,
                            back_liquidity=dnb_sel.available_to_back,
                            lay_liquidity=mo_sel.available_to_lay,
                            result=result,
                        )

    return best_opp


def _build_opportunity_3leg(
    mo_market_id: str,
    dnb_market_id: str,
    event_name: str,
    market_start: Any,
    selection_name: str,
    mo_back_price: Decimal,
    dnb_lay_price: Decimal,
    draw_back_price: Decimal,
    back_liquidity: Decimal,
    lay_liquidity: Decimal,
    draw_liquidity: Decimal,
    result: dict,
) -> Opportunity:
    """Build an Opportunity from evaluate_mo_dnb_3leg_arb result."""
    selections = (
        {
            "selection_id": selection_name,
            "name": selection_name,
            "back_price": float(mo_back_price),
            "lay_price": float(dnb_lay_price),
            "stake_eur": float(result["back_stake"]),
            "lay_stake_eur": float(result["lay_stake"]),
            "liquidity_eur": float(min(back_liquidity, lay_liquidity)),
            "back_market_id": mo_market_id,
            "lay_market_id": dnb_market_id,
            "direction": "back_mo_lay_dnb",
        },
        {
            "selection_id": "draw",
            "name": "The Draw",
            "back_price": float(draw_back_price),
            "stake_eur": float(result["draw_stake"]),
            "liquidity_eur": float(draw_liquidity),
            "back_market_id": mo_market_id,
            "direction": "draw_hedge",
        },
    )

    min_liq = min(back_liquidity, lay_liquidity, draw_liquidity)

    return Opportunity(
        market_id=f"{mo_market_id}+{dnb_market_id}",
        event_name=event_name,
        market_start=market_start,
        arb_type="cross_market",
        selections=selections,
        total_stake_eur=result["total_outlay"],
        overround_raw=mo_back_price / dnb_lay_price,
        gross_profit_eur=result["gross_profit"],
        commission_eur=max(
            result["commission_scenarios"]["x_wins"],
            result["commission_scenarios"]["x_loses"],
            result["commission_scenarios"]["draw"],
        ),
        net_profit_eur=result["net_profit"],
        net_roi_pct=result["roi"],
        liquidity_by_selection=(min(back_liquidity, lay_liquidity), draw_liquidity),
    )


def _build_opportunity(
    mo_market_id: str,
    dnb_market_id: str,
    event_name: str,
    market_start: Any,
    selection_name: str,
    direction: str,
    back_price: Decimal,
    lay_price: Decimal,
    back_liquidity: Decimal,
    lay_liquidity: Decimal,
    result: dict,
) -> Opportunity:
    """Build an Opportunity from evaluate_back_lay_arb result."""
    back_market = mo_market_id if direction == "back_mo_lay_dnb" else dnb_market_id
    lay_market = dnb_market_id if direction == "back_mo_lay_dnb" else mo_market_id

    selections = (
        {
            "selection_id": selection_name,  # use name as ID for cross-market
            "name": selection_name,
            "back_price": float(back_price),
            "lay_price": float(lay_price),
            "stake_eur": float(result["back_stake"]),
            "lay_stake_eur": float(result["lay_stake"]),
            "liquidity_eur": float(min(back_liquidity, lay_liquidity)),
            "back_market_id": back_market,
            "lay_market_id": lay_market,
            "direction": direction,
        },
    )

    return Opportunity(
        market_id=f"{mo_market_id}+{dnb_market_id}",
        event_name=event_name,
        market_start=market_start,
        arb_type="cross_market",
        selections=selections,
        total_stake_eur=result["back_stake"] + result["lay_liability"],
        overround_raw=back_price / lay_price,  # ratio as proxy for edge
        gross_profit_eur=min(result["gross_win"], result["gross_lose"]),
        commission_eur=result["commission_win"] + result["commission_lose"],
        net_profit_eur=result["net_profit"],
        net_roi_pct=result["roi"],
        liquidity_by_selection=(min(back_liquidity, lay_liquidity),),
    )
