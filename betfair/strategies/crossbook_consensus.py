from __future__ import annotations

from typing import Any, Dict, Optional

import config
from core.types import PriceSnapshot


def evaluate_crossbook_consensus(
    *,
    consensus_row: Dict[str, Any],
    snapshot: PriceSnapshot,
    market_meta: Dict[str, Any],
    matched_event: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not getattr(config, "BETFAIR_CROSSBOOK_CONSENSUS_ENABLED", True):
        return None
    source_count = int(consensus_row.get("source_count", 0) or 0)
    confirmation_source_count = max(
        source_count,
        int(matched_event.get("external_source_count", source_count) or source_count),
    )
    match_confidence = float(matched_event.get("match_confidence", 0.0) or 0.0)
    required_sources = int(getattr(config, "BETFAIR_CONSENSUS_MIN_SOURCES", 3))
    if match_confidence >= 0.65:
        required_sources = max(2, required_sources - 1)
    if confirmation_source_count < required_sources:
        return None
    backable = [selection for selection in snapshot.selections if float(selection.best_back_price or 0) > 1.01]
    if len(backable) < 2:
        return None
    favorite = min(backable, key=lambda selection: float(selection.best_back_price))
    betfair_prob = 1.0 / max(1.01, float(favorite.best_back_price))
    consensus_prob = float(consensus_row.get("consensus_prob", 0.0) or 0.0)
    edge = consensus_prob - betfair_prob
    min_edge = 0.03
    dispersion = float(consensus_row.get("consensus_dispersion", 0.0) or 0.0)
    if confirmation_source_count >= 2 and match_confidence >= 0.65:
        min_edge = 0.02
    if dispersion <= 0.015:
        min_edge = min(min_edge, 0.018)
    if abs(edge) < min_edge:
        return None
    source_mix = sorted(
        {
            *[str(item) for item in (consensus_row.get("sources") or []) if str(item).strip()],
            *[str(item) for item in (matched_event.get("source_mix") or []) if str(item).strip()],
        }
    )
    return {
        "strategy_id": "betfair_crossbook_consensus",
        "event_key": str(consensus_row.get("event_key") or snapshot.market_id),
        "event_name": str(market_meta.get("event_name") or snapshot.market_id),
        "market_id": snapshot.market_id,
        "selection_key": str(favorite.selection_id),
        "selection_name": favorite.name,
        "signal_strength": round(abs(edge), 6),
        "expected_edge": round(edge * 100.0, 4),
        "fillability_score": round(max(0.0, min(1.0, float(favorite.available_to_back or 0) / 200.0)), 4),
        "source_mix": source_mix,
        "external_source_count": confirmation_source_count,
        "polymarket_confirmed": bool(matched_event),
        "match_confidence": match_confidence,
        "quote_freshness_sec": 0.0,
        "event_confirmation_level": "high" if abs(edge) >= 0.05 and confirmation_source_count >= 2 else ("medium" if abs(edge) >= min_edge else "low"),
        "expected_half_life_sec": 120,
        "entry_side": "back" if edge > 0 else "lay",
        "entry_back_odds": float(favorite.best_back_price),
        "entry_lay_odds": float(favorite.best_lay_price or 0.0),
        "entry_market_status": str(snapshot.market_status),
        "reason": "consensus_vs_betfair_deviation",
        "strategy_context": {
            "market_type": str(market_meta.get("market_type") or ""),
            "competition": str(market_meta.get("competition_name") or ""),
            "sport": str(market_meta.get("sport_name") or ""),
            "betfair_prob": round(betfair_prob, 6),
            "consensus_prob": round(consensus_prob, 6),
            "consensus_dispersion": float(consensus_row.get("consensus_dispersion", 0.0) or 0.0),
        },
    }
