"""
Deterministic feature extraction for opportunity scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from core.types import MarketMicrostructure, Opportunity, PriceSnapshot


@dataclass(frozen=True)
class FeatureVector:
    """Flat feature schema used by model inference."""
    market_id: str
    selection_count: int
    arb_type: str
    overround_back: Decimal
    overround_lay: Decimal
    min_back_liquidity_eur: Decimal
    min_lay_liquidity_eur: Decimal
    stake_eur: Decimal
    net_profit_eur: Decimal
    net_roi_pct: Decimal
    microstructure: MarketMicrostructure


def _safe_time_to_start_sec(market_start: Optional[datetime]) -> int:
    if market_start is None:
        return 0
    now = datetime.now(timezone.utc)
    if market_start.tzinfo is None:
        market_start = market_start.replace(tzinfo=timezone.utc)
    return int((market_start - now).total_seconds())


def _infer_in_play(time_to_start_sec: int) -> bool:
    return time_to_start_sec <= 0


def _decimal_abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def build_market_microstructure(
    snapshot: PriceSnapshot,
    market_start: Optional[datetime] = None,
    previous_snapshot: Optional[PriceSnapshot] = None,
) -> MarketMicrostructure:
    spreads = []
    total_back = Decimal("0")
    total_lay = Decimal("0")
    max_back = Decimal("0")
    weighted_spread_num = Decimal("0")
    for sel in snapshot.selections:
        total_back += sel.available_to_back
        total_lay += sel.available_to_lay
        if sel.available_to_back > max_back:
            max_back = sel.available_to_back
        if sel.best_back_price > Decimal("0") and sel.best_lay_price > Decimal("0"):
            spread = sel.best_lay_price - sel.best_back_price
            spreads.append(spread)
            weighted_spread_num += spread * sel.available_to_back

    spread_mean = sum(spreads, Decimal("0")) / Decimal(len(spreads)) if spreads else Decimal("0")
    denom = total_back + total_lay
    imbalance = (total_back - total_lay) / denom if denom > Decimal("0") else Decimal("0")
    depth_total = denom
    weighted_spread = (weighted_spread_num / total_back) if total_back > Decimal("0") else Decimal("0")
    lay_back_ratio = (total_lay / total_back) if total_back > Decimal("0") else Decimal("0")
    top_concentration = (max_back / total_back) if total_back > Decimal("0") else Decimal("0")

    price_velocity = Decimal("0")
    short_volatility = Decimal("0")
    if previous_snapshot is not None and len(previous_snapshot.selections) == len(snapshot.selections):
        deltas = []
        for curr, prev in zip(snapshot.selections, previous_snapshot.selections):
            if curr.best_back_price > Decimal("0") and prev.best_back_price > Decimal("0"):
                deltas.append(curr.best_back_price - prev.best_back_price)
        if deltas:
            price_velocity = sum(deltas, Decimal("0")) / Decimal(len(deltas))
            short_volatility = sum((_decimal_abs(d) for d in deltas), Decimal("0")) / Decimal(len(deltas))

    # volume_momentum: change in total depth vs previous snapshot
    volume_momentum = Decimal("0")
    if previous_snapshot is not None:
        prev_total_back = sum(s.available_to_back for s in previous_snapshot.selections)
        prev_total_lay = sum(s.available_to_lay for s in previous_snapshot.selections)
        prev_depth = prev_total_back + prev_total_lay
        volume_momentum = depth_total - prev_depth

    # back_lay_crossover: 1.0 if any selection has best_back_price > best_lay_price (and both > 0)
    back_lay_crossover = Decimal("0")
    for sel in snapshot.selections:
        if sel.best_back_price > Decimal("0") and sel.best_lay_price > Decimal("0"):
            if sel.best_back_price > sel.best_lay_price:
                back_lay_crossover = Decimal("1")
                break

    # overround_distance: abs(back_overround - 1.0)
    back_prices_all = [sel.best_back_price for sel in snapshot.selections if sel.best_back_price > Decimal("0")]
    back_overround = sum((Decimal("1") / p for p in back_prices_all), Decimal("0")) if back_prices_all else Decimal("0")
    overround_distance = _decimal_abs(back_overround - Decimal("1"))

    # depth_ratio_top3: top 3 selections by available_to_back / total_back
    depth_ratio_top3 = Decimal("0")
    if total_back > Decimal("0"):
        sorted_backs = sorted(
            (sel.available_to_back for sel in snapshot.selections),
            reverse=True,
        )
        top3_sum = sum(sorted_backs[:3], Decimal("0"))
        depth_ratio_top3 = top3_sum / total_back

    # price_range: max(back_prices) - min(back_prices) across selections with back_price > 0
    price_range = Decimal("0")
    if back_prices_all:
        price_range = max(back_prices_all) - min(back_prices_all)

    time_to_start_sec = _safe_time_to_start_sec(market_start)
    return MarketMicrostructure(
        spread_mean=spread_mean,
        imbalance=imbalance,
        depth_total_eur=depth_total,
        price_velocity=price_velocity,
        short_volatility=short_volatility,
        time_to_start_sec=time_to_start_sec,
        in_play=_infer_in_play(time_to_start_sec),
        weighted_spread=weighted_spread,
        lay_back_ratio=lay_back_ratio,
        top_of_book_concentration=top_concentration,
        selection_count=len(snapshot.selections),
        volume_momentum=volume_momentum,
        back_lay_crossover=back_lay_crossover,
        overround_distance=overround_distance,
        depth_ratio_top3=depth_ratio_top3,
        price_range=price_range,
    )


def build_feature_vector(
    snapshot: PriceSnapshot,
    opportunity: Opportunity,
    market_start: Optional[datetime] = None,
    previous_snapshot: Optional[PriceSnapshot] = None,
) -> FeatureVector:
    back_prices = [s.best_back_price for s in snapshot.selections if s.best_back_price > Decimal("0")]
    lay_prices = [s.best_lay_price for s in snapshot.selections if s.best_lay_price > Decimal("0")]
    back_overround = sum((Decimal("1") / p for p in back_prices), Decimal("0")) if back_prices else Decimal("0")
    lay_overround = sum((Decimal("1") / p for p in lay_prices), Decimal("0")) if lay_prices else Decimal("0")

    min_back_liq = min((s.available_to_back for s in snapshot.selections), default=Decimal("0"))
    min_lay_liq = min((s.available_to_lay for s in snapshot.selections), default=Decimal("0"))

    return FeatureVector(
        market_id=snapshot.market_id,
        selection_count=len(snapshot.selections),
        arb_type=opportunity.arb_type,
        overround_back=back_overround,
        overround_lay=lay_overround,
        min_back_liquidity_eur=min_back_liq,
        min_lay_liquidity_eur=min_lay_liq,
        stake_eur=opportunity.total_stake_eur,
        net_profit_eur=opportunity.net_profit_eur,
        net_roi_pct=opportunity.net_roi_pct,
        microstructure=build_market_microstructure(
            snapshot=snapshot,
            market_start=market_start,
            previous_snapshot=previous_snapshot,
        ),
    )
