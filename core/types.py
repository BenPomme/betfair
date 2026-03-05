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
    runner_status: str = "UNKNOWN"               # e.g. ACTIVE, WINNER, LOSER


@dataclass
class PriceSnapshot:
    """Latest prices for a market. Used by cache and scanner."""
    market_id: str
    selections: Tuple[SelectionPrice, ...]
    timestamp: datetime  # For staleness check (reject if older than STALE_PRICE_SECONDS)
    market_status: str = "OPEN"  # e.g. OPEN, SUSPENDED, CLOSED


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


@dataclass(frozen=True)
class MarketMicrostructure:
    """Per-market microstructure metrics used for ML scoring."""
    spread_mean: Decimal
    imbalance: Decimal
    depth_total_eur: Decimal
    price_velocity: Decimal
    short_volatility: Decimal
    time_to_start_sec: int
    in_play: bool
    weighted_spread: Decimal = Decimal("0")
    lay_back_ratio: Decimal = Decimal("0")
    top_of_book_concentration: Decimal = Decimal("0")
    selection_count: int = 0
    volume_momentum: Decimal = Decimal("0")
    back_lay_crossover: Decimal = Decimal("0")
    overround_distance: Decimal = Decimal("0")
    depth_ratio_top3: Decimal = Decimal("0")
    price_range: Decimal = Decimal("0")


@dataclass(frozen=True)
class ScoredOpportunity:
    """Scoring output used by execution gating and telemetry."""
    edge_score: Decimal
    fill_prob: Decimal
    expected_net_profit_eur: Decimal
    decision: str  # "EXECUTE", "DEFER", or "SKIP"
    dynamic_threshold_eur: Decimal
    model_version: str
    confidence: Decimal
    order_policy: str  # "best", "improve", "defer"
    ttl_seconds: int
    reason: str
    prediction_influence: str = "none"  # "none", "boosted", "penalized", "ignored_insufficient_data"
    stake_multiplier: Decimal = Decimal("1")
