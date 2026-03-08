from __future__ import annotations

from typing import Any, Dict, Iterable, List


def evaluate_polymarket_binary_research(quotes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Research-only model for binary-option microstructure on Polymarket.

    Philosophy:
    - binary contracts are bounded probability instruments
    - the best research signals are spread, liquidity, and short-horizon repricing pressure
    - this is not an execution strategy yet; it is a ranked candidate monitor
    """

    quote_list = [dict(row) for row in quotes if isinstance(row, dict)]
    if not quote_list:
        return {
            "strategy_id": "polymarket_binary_research",
            "label": "Polymarket Binary Research",
            "mode": "research",
            "explainer": "Studies binary-contract spread, liquidity, and repricing pressure on Polymarket to rank research candidates before any trading logic is trusted.",
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "realized_net_pnl": 0.0,
            "expected_net_edge": 0.0,
            "fillability_avg": 0.0,
            "learning_progress_pct": 0.0,
            "top_blockers": ["awaiting_polymarket_quotes"],
            "latest_candidates": [],
            "research_summary": {
                "tight_spread_quotes": 0,
                "momentum_candidates": 0,
                "mean_reversion_candidates": 0,
                "avg_spread_bps": None,
            },
        }

    latest_candidates: List[Dict[str, Any]] = []
    spread_bps_total = 0.0
    spread_obs = 0
    tight_spread_quotes = 0
    momentum_candidates = 0
    mean_reversion_candidates = 0
    candidate_count = 0
    accepted_count = 0
    expected_edge = 0.0
    fillability_sum = 0.0

    for row in quote_list:
        price = float(row.get("last_trade_price", row.get("probability", 0.0)) or 0.0)
        bid = float(row.get("best_bid", 0.0) or 0.0)
        ask = float(row.get("best_ask", 0.0) or 0.0)
        liquidity = float(row.get("liquidity", 0.0) or 0.0)
        one_day_move = float(row.get("one_day_price_change", 0.0) or 0.0)
        one_week_move = float(row.get("one_week_price_change", 0.0) or 0.0)
        spread = max(0.0, ask - bid) if bid > 0 and ask > 0 else 0.0
        spread_bps = (spread / max(price, 0.01)) * 10000.0 if spread > 0 else 0.0
        if spread_bps > 0:
            spread_bps_total += spread_bps
            spread_obs += 1
        if spread_bps and spread_bps <= 350:
            tight_spread_quotes += 1

        momentum_signal = abs(one_day_move) >= 0.08 and liquidity >= 1000 and 0.1 <= price <= 0.9 and spread_bps <= 400
        mean_reversion_signal = abs(one_week_move) >= 0.15 and spread_bps <= 300 and liquidity >= 1500 and (price <= 0.2 or price >= 0.8)
        if momentum_signal:
            momentum_candidates += 1
        if mean_reversion_signal:
            mean_reversion_candidates += 1
        if not (momentum_signal or mean_reversion_signal):
            continue

        candidate_count += 1
        accepted_count += 1
        fillability = min(1.0, max(0.05, liquidity / 5000.0))
        fillability_sum += fillability
        expected = max(0.0, abs(one_day_move) * 0.35 + abs(one_week_move) * 0.2 - (spread_bps / 10000.0))
        expected_edge += expected
        latest_candidates.append(
            {
                "strategy_id": "polymarket_binary_research",
                "market_id": row.get("market_slug") or row.get("event_slug") or row.get("title"),
                "selection_key": row.get("market_slug") or row.get("title"),
                "event_name": row.get("title"),
                "event_key": row.get("event_slug") or row.get("title"),
                "reason": "momentum" if momentum_signal else "mean_reversion",
                "source_mix": ["polymarket"],
                "signal_strength": round(abs(one_day_move) + abs(one_week_move), 4),
                "expected_edge": round(expected, 4),
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

    latest_candidates = latest_candidates[:8]
    avg_spread_bps = round(spread_bps_total / spread_obs, 2) if spread_obs else None
    progress = min(100.0, round((candidate_count / 50.0) * 100.0, 2))
    blockers: List[str] = []
    if candidate_count == 0:
        blockers.append("no_binary_microstructure_candidates")

    return {
        "strategy_id": "polymarket_binary_research",
        "label": "Polymarket Binary Research",
        "mode": "research",
        "explainer": "Studies binary-contract spread, liquidity, and repricing pressure on Polymarket to rank research candidates before any trading logic is trusted.",
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": 0,
        "realized_net_pnl": 0.0,
        "expected_net_edge": round(expected_edge, 4),
        "fillability_avg": round(fillability_sum / accepted_count, 4) if accepted_count else 0.0,
        "learning_progress_pct": progress,
        "top_blockers": blockers,
        "latest_candidates": latest_candidates,
        "research_summary": {
            "tight_spread_quotes": tight_spread_quotes,
            "momentum_candidates": momentum_candidates,
            "mean_reversion_candidates": mean_reversion_candidates,
            "avg_spread_bps": avg_spread_bps,
        },
    }
