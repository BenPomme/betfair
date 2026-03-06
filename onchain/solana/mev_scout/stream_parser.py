from __future__ import annotations

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
        amount = float(payload.get("amount_usd", 0.0) or 0.0)
        if venue is None and amount <= 0:
            return None
        return {
            "venue": venue or "unknown",
            "amount_usd": amount,
            "wallet": payload.get("wallet"),
            "signature": payload.get("signature"),
        }
