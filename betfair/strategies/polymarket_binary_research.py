from __future__ import annotations

from typing import Any, Dict, Iterable, List


def build_polymarket_binary_candidates(quotes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in [dict(item) for item in quotes if isinstance(item, dict)]:
        price = float(row.get("last_trade_price", row.get("probability", 0.0)) or 0.0)
        bid = float(row.get("best_bid", 0.0) or 0.0)
        ask = float(row.get("best_ask", 0.0) or 0.0)
        liquidity = float(row.get("liquidity", 0.0) or 0.0)
        one_day_move = float(row.get("one_day_price_change", 0.0) or 0.0)
        one_week_move = float(row.get("one_week_price_change", 0.0) or 0.0)
        spread = max(0.0, ask - bid) if bid > 0 and ask > 0 else 0.0
        spread_bps = (spread / max(price, 0.01)) * 10000.0 if spread > 0 else 0.0

        momentum_signal = abs(one_day_move) >= 0.04 and liquidity >= 300 and 0.05 <= price <= 0.95 and spread_bps <= 650
        mean_reversion_signal = abs(one_week_move) >= 0.08 and spread_bps <= 500 and liquidity >= 500 and (price <= 0.25 or price >= 0.75)
        flow_dislocation_signal = abs(one_day_move) >= 0.025 and liquidity >= 250 and spread_bps <= 250 and 0.08 <= price <= 0.92
        if not (momentum_signal or mean_reversion_signal or flow_dislocation_signal):
            continue

        fillability = min(1.0, max(0.05, liquidity / 3000.0))
        heuristic_edge = max(
            0.0,
            abs(one_day_move) * 0.45 + abs(one_week_move) * 0.28 - (spread_bps / 12000.0)
        )
        signal_strength = abs(one_day_move) + abs(one_week_move)
        candidates.append(
            {
                "strategy_id": "polymarket_binary_research",
                "market_id": row.get("market_slug") or row.get("event_slug") or row.get("title"),
                "selection_key": row.get("market_slug") or row.get("title"),
                "event_name": row.get("title"),
                "event_key": row.get("event_slug") or row.get("title"),
                "reason": "momentum" if momentum_signal else ("mean_reversion" if mean_reversion_signal else "flow_dislocation"),
                "source_mix": ["polymarket"],
                "signal_strength": round(signal_strength, 4),
                "expected_edge": round(heuristic_edge, 6),
                "fillability_score": round(fillability, 4),
                "event_confirmation_level": "research",
                "polymarket_confirmed": True,
                "match_confidence": 1.0,
                "quote_freshness_sec": 0.0,
                "expected_half_life_sec": 300 if momentum_signal else 1800,
                "strategy_context": {
                    "sport": row.get("sport"),
                    "liquidity": liquidity,
                    "price": price,
                    "spread_bps": round(spread_bps, 2),
                    "one_day_price_change": one_day_move,
                    "one_week_price_change": one_week_move,
                },
            }
        )
    return candidates


def summarize_polymarket_binary_research(
    quotes: Iterable[Dict[str, Any]],
    candidates: Iterable[Dict[str, Any]],
    *,
    model_state: Dict[str, Any],
    label_state: Dict[str, Any],
) -> Dict[str, Any]:
    quote_list = [dict(row) for row in quotes if isinstance(row, dict)]
    candidate_list = [dict(row) for row in candidates if isinstance(row, dict)]
    spread_bps_values: List[float] = []
    for row in quote_list:
        price = float(row.get("last_trade_price", row.get("probability", 0.0)) or 0.0)
        bid = float(row.get("best_bid", 0.0) or 0.0)
        ask = float(row.get("best_ask", 0.0) or 0.0)
        spread = max(0.0, ask - bid) if bid > 0 and ask > 0 else 0.0
        if spread > 0:
            spread_bps_values.append((spread / max(price, 0.01)) * 10000.0)

    learned_examples = int(model_state.get("labeled_examples", 0) or 0)
    avg_realized_edge = float(label_state.get("avg_realized_edge", 0.0) or 0.0)
    progress = min(100.0, round((learned_examples / 200.0) * 100.0, 2))
    blockers: List[str] = []
    if not candidate_list:
        blockers.append("no_binary_microstructure_candidates")
    if learned_examples < 25:
        blockers.append("insufficient_labeled_examples")

    return {
        "strategy_id": "polymarket_binary_research",
        "label": "Polymarket Binary Research",
        "mode": "research",
        "explainer": "Learns which Polymarket binary setups keep moving or mean-revert after spread and liquidity costs, then ranks new research candidates from real observed outcomes.",
        "candidate_count": len(candidate_list),
        "accepted_count": len(candidate_list),
        "rejected_count": 0,
        "realized_net_pnl": round(float(label_state.get("realized_net_pnl", 0.0) or 0.0), 6),
        "expected_net_edge": round(sum(float(item.get("expected_edge", 0.0) or 0.0) for item in candidate_list), 6),
        "fillability_avg": round(
            sum(float(item.get("fillability_score", 0.0) or 0.0) for item in candidate_list) / max(1, len(candidate_list)),
            4,
        ),
        "learning_progress_pct": progress,
        "top_blockers": blockers,
        "latest_candidates": candidate_list[:8],
        "research_summary": {
            "avg_spread_bps": round(sum(spread_bps_values) / len(spread_bps_values), 2) if spread_bps_values else None,
            "labeled_examples": learned_examples,
            "pending_labels": int(label_state.get("pending_labels", 0) or 0),
            "win_rate": round(float(label_state.get("win_rate", 0.0) or 0.0), 4),
            "avg_realized_edge": round(avg_realized_edge, 6),
            "model_bucket_count": len(model_state.get("buckets") or []),
            "best_bucket": (model_state.get("buckets") or [{}])[0] if model_state.get("buckets") else None,
        },
    }
