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
    if source_count < int(getattr(config, "BETFAIR_CONSENSUS_MIN_SOURCES", 3)):
        return None
    backable = [selection for selection in snapshot.selections if float(selection.best_back_price or 0) > 1.01]
    if len(backable) < 2:
        return None
    favorite = min(backable, key=lambda selection: float(selection.best_back_price))
    betfair_prob = 1.0 / max(1.01, float(favorite.best_back_price))
    consensus_prob = float(consensus_row.get("consensus_prob", 0.0) or 0.0)
    edge = consensus_prob - betfair_prob
    if abs(edge) < 0.03:
        return None
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
        "source_mix": list(consensus_row.get("sources") or []),
        "external_source_count": source_count,
        "polymarket_confirmed": bool(matched_event),
        "match_confidence": float(matched_event.get("match_confidence", 0.0) or 0.0),
        "quote_freshness_sec": 0.0,
        "event_confirmation_level": "medium" if abs(edge) >= 0.05 else "low",
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
