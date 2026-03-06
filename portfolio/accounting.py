from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Mapping, Optional

from portfolio.types import StrategyAccount


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_balance_history(rows: Optional[Iterable]) -> List[dict]:
    result: List[dict] = []
    if not rows:
        return result
    for row in rows:
        if isinstance(row, Mapping):
            ts = row.get("ts") or row.get("timestamp")
            balance = row.get("balance")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            ts, balance = row[0], row[1]
        else:
            continue
        try:
            result.append({"ts": str(ts), "balance": float(balance)})
        except Exception:
            continue
    return result


def compute_drawdown_pct(balance_history: Optional[Iterable]) -> float:
    rows = normalize_balance_history(balance_history)
    peak: Optional[float] = None
    max_drawdown = 0.0
    for row in rows:
        balance = float(row.get("balance", 0.0))
        if peak is None or balance > peak:
            peak = balance
        if peak and peak > 0:
            drawdown = ((peak - balance) / peak) * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    return round(max_drawdown, 4)


def build_strategy_account(
    *,
    portfolio_id: str,
    currency: str,
    starting_balance: float,
    current_balance: float,
    realized_pnl: float,
    unrealized_pnl: float = 0.0,
    fees_paid: float = 0.0,
    slippage_cost: float = 0.0,
    gross_exposure: float = 0.0,
    wins: int = 0,
    losses: int = 0,
    trade_count: int = 0,
    balance_history: Optional[Iterable] = None,
) -> StrategyAccount:
    roi_pct = (realized_pnl / starting_balance * 100.0) if starting_balance else 0.0
    drawdown_pct = compute_drawdown_pct(balance_history)
    return StrategyAccount(
        portfolio_id=portfolio_id,
        currency=currency,
        starting_balance=round(float(starting_balance), 6),
        current_balance=round(float(current_balance), 6),
        realized_pnl=round(float(realized_pnl), 6),
        unrealized_pnl=round(float(unrealized_pnl), 6),
        fees_paid=round(float(fees_paid), 6),
        slippage_cost=round(float(slippage_cost), 6),
        gross_exposure=round(float(gross_exposure), 6),
        roi_pct=round(float(roi_pct), 6),
        drawdown_pct=round(float(drawdown_pct), 6),
        wins=int(wins),
        losses=int(losses),
        trade_count=int(trade_count),
        last_updated=utc_now_iso(),
    )
