from __future__ import annotations

from typing import Any, Dict, Iterable, List

from polymarket.utils import clamp, parse_ts, rolling_mean, rolling_std, to_float, utc_now


def _history_metrics(history: Iterable[Dict[str, Any]], current_price: float) -> Dict[str, float]:
    points = [dict(item) for item in history if isinstance(item, dict)]
    if not points:
        return {
            "return_2m": 0.0,
            "return_10m": 0.0,
            "return_30m": 0.0,
            "realized_vol": 0.0,
            "history_depth": 0,
        }
    prices = [clamp(to_float(item.get("price"), current_price)) for item in points]
    latest = current_price if current_price > 0 else prices[-1]

    def _return(index_from_end: int) -> float:
        if len(prices) <= index_from_end:
            base = prices[0]
        else:
            base = prices[-1 - index_from_end]
        if base <= 0:
            return 0.0
        return latest - base

    deltas = [prices[idx] - prices[idx - 1] for idx in range(1, len(prices))]
    return {
        "return_2m": round(_return(2), 6),
        "return_10m": round(_return(10), 6),
        "return_30m": round(_return(30), 6),
        "realized_vol": round(rolling_std(deltas), 6),
        "history_depth": len(points),
    }


def build_feature_rows(
    token_rows: Iterable[Dict[str, Any]],
    *,
    histories: Dict[str, List[Dict[str, Any]]] | None = None,
    stale_quote_seconds: int = 30,
    fee_bps: float = 0.0,
    queue_penalty_bps: float = 0.0,
) -> List[Dict[str, Any]]:
    history_map = histories or {}
    rows = [dict(item) for item in token_rows if isinstance(item, dict)]
    counterpart_prices: Dict[str, List[float]] = {}
    for row in rows:
        market_key = str(row.get("market_slug") or row.get("market_id") or "")
        counterpart_prices.setdefault(market_key, []).append(
            to_float(row.get("midpoint") or row.get("last_trade_price") or row.get("gamma_price"), 0.0)
        )

    features: List[Dict[str, Any]] = []
    now = utc_now()
    for row in rows:
        token_id = str(row.get("token_id") or "")
        market_key = str(row.get("market_slug") or row.get("market_id") or "")
        price = clamp(to_float(row.get("midpoint") or row.get("last_trade_price") or row.get("gamma_price"), 0.0))
        gamma_price = clamp(to_float(row.get("gamma_price"), price))
        best_bid = clamp(to_float(row.get("best_bid"), 0.0))
        best_ask = clamp(to_float(row.get("best_ask"), 0.0))
        spread = max(0.0, best_ask - best_bid) if best_bid > 0 and best_ask > 0 else 0.0
        spread_bps = (spread / max(price, 0.01)) * 10000.0 if spread else 0.0
        bid_depth = max(0.0, to_float(row.get("bid_depth"), 0.0))
        ask_depth = max(0.0, to_float(row.get("ask_depth"), 0.0))
        depth_total = bid_depth + ask_depth
        imbalance = ((bid_depth - ask_depth) / depth_total) if depth_total > 0 else 0.0
        quote_ts = parse_ts(row.get("book_timestamp") or row.get("timestamp") or row.get("observed_at"))
        quote_freshness_sec = (now - quote_ts).total_seconds() if quote_ts is not None else float(stale_quote_seconds)
        siblings = counterpart_prices.get(market_key) or []
        complement_gap = round(sum(siblings) - 1.0, 6) if len(siblings) >= 2 else 0.0
        history_metrics = _history_metrics(history_map.get(token_id) or [], price)
        gamma_delta = round(price - gamma_price, 6)
        last_trade_delta = round(to_float(row.get("last_trade_price"), price) - price, 6)
        coherence_score = clamp(
            1.0
            - (
                min(abs(gamma_delta) * 2.0, 1.0) * 0.45
                + min(abs(complement_gap) * 2.0, 1.0) * 0.35
                + min(abs(last_trade_delta) * 2.0, 1.0) * 0.20
            )
        )
        interference = round(
            abs(history_metrics["return_2m"] - history_metrics["return_10m"])
            + abs(imbalance) * 0.25
            + history_metrics["realized_vol"],
            6,
        )
        energy = round(
            abs(gamma_delta) * 1.4
            + abs(complement_gap) * 1.1
            + abs(history_metrics["return_2m"]) * 0.9
            + history_metrics["realized_vol"] * 1.3
            + (spread_bps / 10000.0),
            6,
        )
        basin_depth = round((depth_total / max(energy * 2000.0, 1.0)), 6)
        relaxation_speed = round(
            abs(history_metrics["return_2m"]) / max(history_metrics["realized_vol"], 0.0005),
            6,
        )
        depth_score = min(1.0, depth_total / 8000.0)
        spread_score = max(0.0, 1.0 - (spread_bps / 400.0))
        freshness_score = max(0.0, 1.0 - (quote_freshness_sec / max(1, stale_quote_seconds)))
        folding_confidence = round(
            clamp((coherence_score * 0.45) + (depth_score * 0.2) + (spread_score * 0.2) + (freshness_score * 0.15)),
            6,
        )
        momentum_bias = round((history_metrics["return_2m"] * 0.6) + (history_metrics["return_10m"] * 0.3) + (imbalance * 0.1), 6)
        heuristic_edge = round(
            (coherence_score - 0.5) * 0.10
            + (folding_confidence - 0.5) * 0.08
            + (momentum_bias * 0.35)
            - (spread_bps / 10000.0)
            - (fee_bps / 10000.0)
            - (queue_penalty_bps / 10000.0),
            6,
        )
        features.append(
            {
                **row,
                **history_metrics,
                "price": price,
                "spread": round(spread, 6),
                "spread_bps": round(spread_bps, 4),
                "quote_freshness_sec": round(quote_freshness_sec, 3),
                "gamma_delta": gamma_delta,
                "last_trade_delta": last_trade_delta,
                "coherence_score": round(coherence_score, 6),
                "interference_score": interference,
                "energy_proxy": energy,
                "basin_depth": basin_depth,
                "relaxation_speed": relaxation_speed,
                "folding_confidence": folding_confidence,
                "orderbook_imbalance": round(imbalance, 6),
                "depth_total_usd": round(depth_total, 6),
                "cost_buffer": round((spread_bps + fee_bps + queue_penalty_bps) / 10000.0, 6),
                "momentum_bias": momentum_bias,
                "heuristic_edge": heuristic_edge,
            }
        )
    return features


def feature_vector(row: Dict[str, Any], feature_names: Iterable[str]) -> List[float]:
    vector: List[float] = []
    for name in feature_names:
        value = to_float(row.get(name), 0.0)
        vector.append(max(-5.0, min(5.0, value)))
    return vector


def summarize_feature_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    points = [dict(item) for item in rows if isinstance(item, dict)]
    freshness = [to_float(item.get("quote_freshness_sec"), 0.0) for item in points]
    folding = [to_float(item.get("folding_confidence"), 0.0) for item in points]
    energy = [to_float(item.get("energy_proxy"), 0.0) for item in points]
    return {
        "quote_count": len(points),
        "avg_quote_freshness_sec": round(rolling_mean(freshness), 3) if freshness else None,
        "avg_folding_confidence": round(rolling_mean(folding), 6) if folding else None,
        "avg_energy_proxy": round(rolling_mean(energy), 6) if energy else None,
    }
