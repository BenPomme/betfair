from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.types import PriceSnapshot


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def evaluate_timezone_decay(
    *,
    snapshot: PriceSnapshot,
    market_meta: Dict[str, Any],
    stale_snapshot_ratio: float,
) -> Optional[Dict[str, Any]]:
    backable = [selection for selection in snapshot.selections if float(selection.best_back_price or 0) > 1.01]
    if len(backable) < 2:
        return None
    market_start = _parse_utc(market_meta.get("market_start"))
    now = datetime.now(timezone.utc)
    local_hour = now.hour
    if market_start is not None:
        local_hour = market_start.astimezone(timezone.utc).hour
    low_attention = 1.0 if local_hour < 8 or local_hour >= 22 else 0.35
    liquidity = sum(float(selection.available_to_back or 0) for selection in backable)
    if liquidity <= 0:
        return None
    viable = []
    for selection in backable:
        back_price = float(selection.best_back_price or 0.0)
        lay_price = float(selection.best_lay_price or 0.0)
        available = float(selection.available_to_back or 0.0)
        spread = (lay_price - back_price) if lay_price > 1.01 else 0.0
        if available < 20.0:
            continue
        if spread > 0.22:
            continue
        viable.append(selection)
    if not viable:
        return None
    favorite = max(
        viable,
        key=lambda selection: (
            float(selection.available_to_back or 0.0) / max(float(selection.best_back_price or 1.01), 1.01),
            -float(selection.best_back_price or 0.0),
        ),
    )
    score = (low_attention * 0.42) + min(0.33, stale_snapshot_ratio * 0.45) + min(0.25, 100.0 / liquidity)
    if score < 0.32:
        return None
    return {
        "strategy_id": "betfair_timezone_decay",
        "event_key": str(market_meta.get("event_id") or market_meta.get("event_name") or snapshot.market_id),
        "event_name": str(market_meta.get("event_name") or snapshot.market_id),
        "market_id": snapshot.market_id,
        "selection_key": str(favorite.selection_id),
        "selection_name": favorite.name,
        "signal_strength": round(score, 6),
        "expected_edge": round(score * 40.0, 4),
        "fillability_score": round(max(0.0, min(1.0, float(favorite.available_to_back or 0) / 150.0)), 4),
        "source_mix": ["betfair_market_ops"],
        "external_source_count": 0,
        "polymarket_confirmed": False,
        "match_confidence": 0.0,
        "quote_freshness_sec": None,
        "event_confirmation_level": "medium" if score >= 0.55 else "low",
        "expected_half_life_sec": 180,
        "entry_side": "back",
        "entry_back_odds": float(favorite.best_back_price),
        "entry_lay_odds": float(favorite.best_lay_price or 0.0),
        "entry_market_status": str(snapshot.market_status),
        "reason": "low_attention_market_ops_decay",
        "strategy_context": {
            "market_type": str(market_meta.get("market_type") or ""),
            "competition": str(market_meta.get("competition_name") or ""),
            "sport": str(market_meta.get("sport_name") or ""),
            "local_hour": local_hour,
            "liquidity": round(liquidity, 2),
            "stale_snapshot_ratio": round(stale_snapshot_ratio, 4),
        },
    }
