from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional


class SolanaStreamParser:
    DEX_MARKERS = {
        "raydium": "raydium",
        "orca": "orca",
        "jupiter": "jupiter",
    }

    def parse(self, payload: Dict) -> Optional[Dict]:
        if not payload:
            return None
        text = str(payload)
        venue = next((name for marker, name in self.DEX_MARKERS.items() if marker in text.lower()), None)
        amount = float(payload.get("amount_usd", payload.get("notional_usd", 0.0)) or 0.0)
        if venue is None and amount <= 0:
            return None
        return {
            "venue": venue or "unknown",
            "amount_usd": amount,
            "wallet": payload.get("wallet"),
            "signature": payload.get("signature"),
            "route": payload.get("route") or payload.get("route_hint"),
            "route_hops": int(payload.get("route_hops", payload.get("hop_count", 1)) or 1),
            "latency_hint_ms": float(payload.get("latency_ms", payload.get("latency_hint_ms", 0.0)) or 0.0),
            "realized_edge_usd": payload.get("realized_edge_usd"),
            "realized_edge_bps": payload.get("realized_edge_bps"),
            "direction": payload.get("direction"),
            "ts": payload.get("ts") or datetime.now(timezone.utc).isoformat(),
        }
