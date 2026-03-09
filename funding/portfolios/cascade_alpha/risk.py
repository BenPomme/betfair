from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import config


class CascadeRiskManager:
    def __init__(self) -> None:
        self._last_trade_at: Dict[str, datetime] = {}

    def can_open(
        self,
        symbol: str,
        open_positions: List[dict],
        gross_exposure: float,
        realized_pnl: float,
        *,
        max_open_positions: int | None = None,
        max_gross_exposure: float | None = None,
    ) -> tuple[bool, str]:
        now = datetime.now(timezone.utc)
        effective_max_open = int(max_open_positions if max_open_positions is not None else config.CASCADE_ALPHA_MAX_OPEN_POSITIONS)
        effective_gross_cap = float(max_gross_exposure if max_gross_exposure is not None else config.CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD)
        if len(open_positions) >= effective_max_open:
            return False, "max_open_positions"
        if gross_exposure >= effective_gross_cap:
            return False, "max_gross_exposure"
        cooldown = int(config.CASCADE_ALPHA_EVENT_COOLDOWN_SECONDS)
        last_trade = self._last_trade_at.get(symbol)
        if last_trade and (now - last_trade).total_seconds() < cooldown:
            return False, "symbol_cooldown"
        daily_loss_limit = float(config.CASCADE_ALPHA_INITIAL_BALANCE_USD) * float(config.CASCADE_ALPHA_DAILY_LOSS_LIMIT_PCT)
        if realized_pnl <= -daily_loss_limit:
            return False, "daily_loss_limit"
        return True, "approved"

    def mark_open(self, symbol: str) -> None:
        self._last_trade_at[symbol] = datetime.now(timezone.utc)

    def should_force_close(self, position: dict, current_price: float) -> tuple[bool, str]:
        entry = float(position.get("entry_price", 0.0) or 0.0)
        if entry <= 0:
            return True, "invalid_entry"
        side = str(position.get("side", "LONG")).upper()
        move = ((current_price - entry) / entry) * (1.0 if side == "LONG" else -1.0)
        hold_seconds = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(position["opened_at"])).total_seconds())
        if hold_seconds >= min(int(config.CASCADE_ALPHA_MAX_HOLD_SECONDS), 600):
            return True, "max_hold"
        if hold_seconds >= 300 and abs(move) < 0.002:
            return True, "stale_setup"
        if move >= 0.009:
            return True, "take_profit"
        if move <= -0.006:
            return True, "stop_loss"
        return False, "hold"
