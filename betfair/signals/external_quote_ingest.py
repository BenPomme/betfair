from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


def build_consensus(quotes: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        event_key = str(quote.get("event_key") or quote.get("event_slug") or "")
        market_type = str(quote.get("market_type") or "moneyline")
        selection_key = str(quote.get("selection_key") or quote.get("market_slug") or quote.get("title") or "")
        if not event_key or not selection_key:
            continue
        buckets[f"{event_key}|{market_type}|{selection_key}"].append(quote)

    consensus: Dict[str, Dict[str, Any]] = {}
    for key, items in buckets.items():
        probabilities = [float(item.get("probability", item.get("last_trade_price", 0.0)) or 0.0) for item in items]
        if not probabilities:
            continue
        avg = sum(probabilities) / len(probabilities)
        variance = sum((value - avg) ** 2 for value in probabilities) / max(1, len(probabilities))
        consensus[key] = {
            "event_key": items[0].get("event_key") or items[0].get("event_slug"),
            "market_type": items[0].get("market_type") or "moneyline",
            "selection_key": items[0].get("selection_key") or items[0].get("market_slug") or items[0].get("title"),
            "consensus_prob": round(avg, 6),
            "source_count": len(items),
            "consensus_dispersion": round(variance ** 0.5, 6),
            "sources": sorted({str(item.get("source") or item.get("provider") or "unknown") for item in items}),
        }
    return consensus
