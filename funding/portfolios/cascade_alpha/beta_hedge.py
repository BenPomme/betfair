from __future__ import annotations

from typing import Dict


def suggest_beta_hedge(symbol: str) -> Dict[str, float | str]:
    symbol = symbol.upper()
    if symbol.startswith("BTC"):
        return {"hedge_symbol": "BTCUSDT", "hedge_ratio": 0.0}
    if symbol.startswith("ETH"):
        return {"hedge_symbol": "ETHUSDT", "hedge_ratio": 0.0}
    return {"hedge_symbol": "BTCUSDT", "hedge_ratio": 0.35}
