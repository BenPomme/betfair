"""
Shared types for the pipeline. Used by scanner, execution, and price cache.
Matches the paper log schema in the brief for Opportunity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class SelectionPrice:
    """Best back/lay price and liquidity for one selection."""
    selection_id: str
    name: str
    best_back_price: Decimal
    available_to_back: Decimal  # EUR
    best_lay_price: Decimal = Decimal("0")      # Decimal("0") if no lay available
    available_to_lay: Decimal = Decimal("0")     # Decimal("0") if no lay available


@dataclass
class PriceSnapshot:
    """Latest prices for a market. Used by cache and scanner."""
    market_id: str
    selections: Tuple[SelectionPrice, ...]
    timestamp: datetime  # For staleness check (reject if older than STALE_PRICE_SECONDS)


@dataclass
class Opportunity:
    """An arb opportunity ready for paper or live execution. Matches paper log schema."""
    market_id: str
    event_name: str
    market_start: Optional[datetime]
    arb_type: str  # "back_back", "lay_lay", or "cross_market"
    selections: Tuple[dict, ...]  # name, back_price/lay_price, stake_eur, liquidity_eur
    total_stake_eur: Decimal
    overround_raw: Decimal
    gross_profit_eur: Decimal
    commission_eur: Decimal
    net_profit_eur: Decimal
    net_roi_pct: Decimal
    liquidity_by_selection: Tuple[Decimal, ...]  # liquidity_eur per selection, same order as selections
