from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from core.types import PriceSnapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


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


def evaluate_suspension_lag(
    *,
    matched_event: Dict[str, Any],
    snapshot: PriceSnapshot,
    market_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not getattr(config, "BETFAIR_SUSPENSION_LAG_ENABLED", True):
        return None
    if str(getattr(snapshot, "market_status", "OPEN") or "OPEN").upper() != "OPEN":
        return None
    backable = [selection for selection in snapshot.selections if float(selection.best_back_price or 0) > 1.01]
    if len(backable) < 2:
        return None
    favorite = min(backable, key=lambda selection: float(selection.best_back_price))
    implied_prob = 1.0 / max(1.01, float(favorite.best_back_price))
    external_prob = float(
        matched_event.get("probability")
        or matched_event.get("last_trade_price")
        or matched_event.get("source_confidence")
        or 0.0
    )
    if external_prob <= 0:
        return None
    signal_strength = abs(external_prob - implied_prob)
    spread = 0.0
    if float(favorite.best_lay_price or 0) > 1.01:
        spread = float(favorite.best_lay_price) - float(favorite.best_back_price)
    in_play = False
    market_start = _parse_utc(market_meta.get("market_start"))
    if market_start is not None:
        in_play = market_start <= _now()
    if in_play:
        signal_strength += 0.02
    if spread > 0.15:
        signal_strength -= 0.03
    source_mix = list(matched_event.get("source_mix") or ["polymarket", "betfair_suspend_resume"])
    external_source_count = int(matched_event.get("external_source_count", 1) or 1)
    dynamic_min_signal = 0.04
    if external_source_count >= 2 or in_play:
        dynamic_min_signal = max(0.025, float(config.BETFAIR_SUSPENSION_LAG_MIN_SIGNAL_STRENGTH) * 0.45)
    elif float(matched_event.get("match_confidence", 0.0) or 0.0) >= 0.75:
        dynamic_min_signal = max(0.03, float(config.BETFAIR_SUSPENSION_LAG_MIN_SIGNAL_STRENGTH) * 0.5)
    confidence = "low"
    if signal_strength >= dynamic_min_signal:
        confidence = "medium"
    if signal_strength >= (float(config.BETFAIR_SUSPENSION_LAG_MIN_SIGNAL_STRENGTH) + 0.08) or external_source_count >= 2:
        confidence = "high"
    if signal_strength < dynamic_min_signal:
        return None
    match_confidence = float(matched_event.get("match_confidence", 0.0) or 0.0)
    quote_freshness_sec = float(matched_event.get("quote_freshness_sec", 0.0) or 0.0)
    return {
        "strategy_id": "betfair_suspension_lag",
        "event_key": str(matched_event.get("external_event_key") or matched_event.get("event_slug") or snapshot.market_id),
        "event_name": str(market_meta.get("event_name") or matched_event.get("title") or snapshot.market_id),
        "market_id": snapshot.market_id,
        "selection_key": str(favorite.selection_id),
        "selection_name": favorite.name,
        "signal_strength": round(signal_strength, 6),
        "expected_edge": round(signal_strength * 100.0, 4),
        "fillability_score": round(max(0.0, min(1.0, float(favorite.available_to_back or 0) / 200.0)), 4),
        "source_mix": source_mix,
        "external_source_count": external_source_count,
        "polymarket_confirmed": bool(
            matched_event.get(
                "polymarket_confirmed",
                ("probability" in matched_event) or ("last_trade_price" in matched_event),
            )
        ),
        "match_confidence": round(match_confidence, 4),
        "quote_freshness_sec": quote_freshness_sec,
        "event_confirmation_level": confidence,
        "expected_half_life_sec": 15 if in_play else 60,
        "entry_side": "back",
        "entry_back_odds": float(favorite.best_back_price),
        "entry_lay_odds": float(favorite.best_lay_price or 0.0),
        "entry_market_status": str(snapshot.market_status),
        "reason": "multi_source_confirmation_vs_betfair_favorite" if external_source_count >= 2 else "polymarket_move_vs_betfair_favorite",
        "strategy_context": {
            "market_type": str(market_meta.get("market_type") or ""),
            "competition": str(market_meta.get("competition_name") or ""),
            "sport": str(market_meta.get("sport_name") or ""),
            "in_play": in_play,
            "favorite_implied_prob": round(implied_prob, 6),
            "external_prob": round(external_prob, 6),
            "spread": round(spread, 6),
            "confirmation_sources": source_mix,
        },
    }
