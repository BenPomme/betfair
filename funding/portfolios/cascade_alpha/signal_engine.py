from __future__ import annotations

from typing import Dict, Optional

import config


class CascadeSignalEngine:
    def classify(self, symbol: str, features: Dict[str, float]) -> Optional[dict]:
        spread_bps = float(features.get("spread_bps", 0.0) or 0.0)
        depth_usd = float(features.get("depth_usd", 0.0) or 0.0)
        if spread_bps > float(config.CASCADE_ALPHA_MAX_SPREAD_BPS) or depth_usd <= 0:
            return None
        price_move_1m = float(features.get("price_return_1m_pct", 0.0) or 0.0)
        price_move_5m = float(features.get("price_return_5m_pct", 0.0) or 0.0)
        oi_change_5m = float(features.get("open_interest_change_5m_pct", 0.0) or 0.0)
        taker_imbalance = float(features.get("taker_imbalance", 0.0) or 0.0)
        liquidation_burst = float(features.get("liquidation_burst_usd", 0.0) or 0.0)
        reference_price = float(features.get("mark_price", 0.0) or 0.0)
        if reference_price <= 0:
            return None
        continuation_score = abs(price_move_1m) + abs(price_move_5m) + max(0.0, oi_change_5m) + abs(taker_imbalance) * 10.0 + (liquidation_burst / 100000.0)
        snapback_score = abs(price_move_5m) + max(0.0, -oi_change_5m) + abs(taker_imbalance) * 5.0 + (liquidation_burst / 150000.0)
        if oi_change_5m < 0 and snapback_score >= float(config.CASCADE_ALPHA_MIN_DEPTH_COLLAPSE_Z):
            return {
                "symbol": symbol,
                "setup": "SNAPBACK",
                "side": "SHORT" if price_move_5m >= 0 else "LONG",
                "signal_score": round(snapback_score, 6),
                "reference_price": reference_price,
                "spread_bps": spread_bps,
                "slippage_bps": min(float(config.CASCADE_ALPHA_MAX_SLIPPAGE_BPS), spread_bps * 0.5 + 6.0),
                "taker_imbalance": taker_imbalance,
                "liquidation_burst_usd": liquidation_burst,
            }
        if continuation_score >= float(config.CASCADE_ALPHA_MIN_LIQUIDATION_Z):
            return {
                "symbol": symbol,
                "setup": "CONTINUATION",
                "side": "LONG" if price_move_1m >= 0 else "SHORT",
                "signal_score": round(continuation_score, 6),
                "reference_price": reference_price,
                "spread_bps": spread_bps,
                "slippage_bps": min(float(config.CASCADE_ALPHA_MAX_SLIPPAGE_BPS), spread_bps * 0.5 + 4.0),
                "taker_imbalance": taker_imbalance,
                "liquidation_burst_usd": liquidation_burst,
            }
        if snapback_score >= float(config.CASCADE_ALPHA_MIN_DEPTH_COLLAPSE_Z):
            return {
                "symbol": symbol,
                "setup": "SNAPBACK",
                "side": "SHORT" if price_move_5m >= 0 else "LONG",
                "signal_score": round(snapback_score, 6),
                "reference_price": reference_price,
                "spread_bps": spread_bps,
                "slippage_bps": min(float(config.CASCADE_ALPHA_MAX_SLIPPAGE_BPS), spread_bps * 0.5 + 6.0),
                "taker_imbalance": taker_imbalance,
                "liquidation_burst_usd": liquidation_burst,
            }
        return None
