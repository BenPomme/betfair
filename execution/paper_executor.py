"""
Paper executor: log opportunity, simulate fill at best and at 1 tick worse,
track virtual balance and P&L. Same interface as live executor (accept Opportunity).
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Any
import json

from core.types import Opportunity
from core import commission as commission_module
import config

# Betfair tick size: 0.01 for prices 2.0-3.0, 0.02 for 1.01-2.0, etc. Simplified: 1 tick = 0.01
TICK_SIZE = Decimal("0.01")


def _tick_worse_back(price: Decimal) -> Decimal:
    """One Betfair tick worse for back (higher decimal odds = worse for back)."""
    return price + TICK_SIZE


def _tick_worse_lay(price: Decimal) -> Decimal:
    """One Betfair tick worse for lay (lower lay price = worse for layer)."""
    return max(price - TICK_SIZE, Decimal("1.01"))


def _realistic_net_profit(opportunity: Opportunity) -> Decimal:
    """Net profit if all legs fill at 1 tick worse than best."""
    if opportunity.arb_type == "lay_lay":
        return _realistic_net_profit_lay(opportunity)
    if opportunity.arb_type == "cross_market":
        return _realistic_net_profit_cross_market(opportunity)
    prices_worse = [_tick_worse_back(Decimal(str(s["back_price"]))) for s in opportunity.selections]
    total_stake = opportunity.total_stake_eur
    result = commission_module.evaluate_back_back_arb(
        prices_worse,
        total_stake,
        config.MBR,
        config.DISCOUNT_RATE,
    )
    if result is None:
        return Decimal("0")
    return result["min_net_profit"]


def _realistic_net_profit_cross_market(opportunity: Opportunity) -> Decimal:
    """Net profit for cross-market if all legs fill at 1 tick worse."""
    sel = opportunity.selections[0]
    direction = sel.get("direction", "")

    # 3-leg trade: back MO + lay DNB + back Draw hedge
    if direction == "back_mo_lay_dnb" and len(opportunity.selections) == 2:
        draw_sel = opportunity.selections[1]
        back_worse = _tick_worse_back(Decimal(str(sel["back_price"])))
        lay_worse = _tick_worse_lay(Decimal(str(sel["lay_price"])))
        draw_worse = _tick_worse_back(Decimal(str(draw_sel["back_price"])))
        result = commission_module.evaluate_mo_dnb_3leg_arb(
            mo_back_price=back_worse,
            dnb_lay_price=lay_worse,
            draw_back_price=draw_worse,
            max_stake=Decimal(str(sel["stake_eur"])),
            mbr=config.MBR,
            discount=config.DISCOUNT_RATE,
        )
        if result is None:
            return Decimal("0")
        return result["net_profit"]

    # 2-leg trade: back DNB + lay MO (direction 2, draw is best scenario)
    back_worse = _tick_worse_back(Decimal(str(sel["back_price"])))
    lay_worse = _tick_worse_lay(Decimal(str(sel["lay_price"])))
    result = commission_module.evaluate_back_lay_arb(
        back_worse,
        lay_worse,
        Decimal(str(sel["stake_eur"])),
        config.MBR,
        config.DISCOUNT_RATE,
    )
    if result is None:
        return Decimal("0")
    return result["net_profit"]


def _realistic_net_profit_lay(opportunity: Opportunity) -> Decimal:
    """Net profit for lay-lay if all legs fill at 1 tick worse (lower lay price)."""
    prices_worse = [_tick_worse_lay(Decimal(str(s["lay_price"]))) for s in opportunity.selections]
    total_stake = opportunity.total_stake_eur
    result = commission_module.evaluate_lay_lay_arb(
        prices_worse,
        total_stake,
        config.MBR,
        config.DISCOUNT_RATE,
    )
    if result is None:
        return Decimal("0")
    return result["min_net_profit"]


def _log_entry(opportunity: Opportunity) -> dict:
    """Build paper log entry matching brief schema."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    market_start_str = None
    if opportunity.market_start:
        if hasattr(opportunity.market_start, "strftime"):
            market_start_str = opportunity.market_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            market_start_str = str(opportunity.market_start)

    is_lay = opportunity.arb_type == "lay_lay"
    is_cross = opportunity.arb_type == "cross_market"

    selections_log = []
    for s in opportunity.selections:
        sel_entry = {"name": s["name"], "stake_eur": s["stake_eur"]}
        if is_cross:
            sel_entry["back_price"] = s.get("back_price")
            sel_entry["lay_price"] = s.get("lay_price")
            sel_entry["lay_stake_eur"] = s.get("lay_stake_eur")
            sel_entry["back_market_id"] = s.get("back_market_id")
            sel_entry["lay_market_id"] = s.get("lay_market_id")
            sel_entry["direction"] = s.get("direction")
        elif is_lay:
            sel_entry["lay_price"] = s["lay_price"]
            sel_entry["liability_eur"] = s.get("liability_eur", 0)
        else:
            sel_entry["back_price"] = s["back_price"]
        selections_log.append(sel_entry)

    liquidity_keys = [f"liquidity_{chr(97 + i)}_eur" for i in range(len(opportunity.liquidity_by_selection))]
    liquidity_dict = dict(zip(liquidity_keys, [float(x) for x in opportunity.liquidity_by_selection]))

    realistic_net = _realistic_net_profit(opportunity)

    entry = {
        "ts": ts,
        "mode": "paper",
        "arb_type": opportunity.arb_type,
        "market_id": opportunity.market_id,
        "event": opportunity.event_name,
        "market_start": market_start_str,
        "selections": selections_log,
        "total_stake_eur": float(opportunity.total_stake_eur),
        "overround_raw": float(opportunity.overround_raw),
        "gross_profit_eur": float(opportunity.gross_profit_eur),
        "commission_eur": float(opportunity.commission_eur),
        "net_profit_eur": float(opportunity.net_profit_eur),
        "net_roi_pct": float(opportunity.net_roi_pct),
        **liquidity_dict,
        "fill_simulated_optimistic": True,
        "fill_simulated_realistic_net": float(realistic_net),
    }
    return entry


class PaperExecutor:
    """Logs opportunities and tracks virtual P&L."""

    def __init__(self, initial_balance_eur: Optional[Decimal] = None):
        self._balance = initial_balance_eur or Decimal("0")
        self._log: List[dict] = []
        self._open_bets = 0

    def log(self, opportunity: Opportunity) -> dict:
        """
        Log the opportunity with full schema; simulate fill at best and 1 tick worse.
        Returns the log entry dict.
        """
        entry = _log_entry(opportunity)
        self._log.append(entry)
        self._balance -= opportunity.total_stake_eur
        self._balance += opportunity.total_stake_eur + opportunity.net_profit_eur
        self._open_bets += 1
        return entry

    def register_settlement(self, net_pnl_eur: Decimal) -> None:
        """When a simulated bet settles (for optional outcome_at_settlement flow)."""
        self._open_bets = max(0, self._open_bets - 1)
        # P&L already applied in log(); no double-count

    @property
    def balance(self) -> Decimal:
        return self._balance

    @property
    def log_entries(self) -> List[dict]:
        return self._log.copy()

    def get_log_as_json(self) -> str:
        return json.dumps(self._log, indent=2)
