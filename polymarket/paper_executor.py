from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import config
from polymarket.utils import clamp, parse_ts, to_float, utc_now_iso


def _walk_levels(
    levels: Iterable[Dict[str, Any]],
    *,
    desired_shares: float,
    fallback_price: float,
    tick_size: float,
    queue_penalty_bps: float,
    side: str,
) -> Tuple[float, float]:
    rows = [dict(item) for item in levels if isinstance(item, dict)]
    if desired_shares <= 0:
        return float(fallback_price), 0.0
    cost = 0.0
    filled = 0.0
    last_price = float(fallback_price)
    last_size = max(1.0, desired_shares)
    for row in rows:
        price = clamp(to_float(row.get("price"), fallback_price), 0.0, 0.999)
        size = max(0.0, to_float(row.get("size"), 0.0))
        if size <= 0:
            continue
        trade_size = min(desired_shares - filled, size)
        cost += trade_size * price
        filled += trade_size
        last_price = price
        last_size = size
        if filled >= desired_shares:
            break
    if filled < desired_shares:
        penalty_steps = max(1.0, (desired_shares - filled) / max(last_size, 1e-6))
        if side.upper() == "BUY":
            penalty_price = min(0.999, last_price + (tick_size * penalty_steps))
        else:
            penalty_price = max(0.001, last_price - (tick_size * penalty_steps))
        cost += (desired_shares - filled) * penalty_price
        filled = desired_shares
    fill_price = (cost / filled) if filled else float(fallback_price)
    if side.upper() == "BUY":
        fill_price *= 1.0 + (queue_penalty_bps / 10000.0)
    else:
        fill_price *= max(0.0, 1.0 - (queue_penalty_bps / 10000.0))
    slippage_bps = 0.0
    if fallback_price > 0:
        if side.upper() == "BUY":
            slippage_bps = ((fill_price - fallback_price) / fallback_price) * 10000.0
        else:
            slippage_bps = ((fallback_price - fill_price) / fallback_price) * 10000.0
    return round(fill_price, 6), round(max(0.0, slippage_bps), 6)


class PolymarketPaperExecutor:
    def __init__(
        self,
        *,
        starting_balance: float,
        fee_bps: float,
        queue_penalty_bps: float,
        max_open_positions: int,
        max_notional_per_trade: float,
        max_positions_per_event: int,
        drawdown_halt_pct: float,
    ) -> None:
        self.starting_balance = float(starting_balance)
        self.fee_bps = float(fee_bps)
        self.queue_penalty_bps = float(queue_penalty_bps)
        self.max_open_positions = int(max_open_positions)
        self.max_notional_per_trade = float(max_notional_per_trade)
        self.max_positions_per_event = int(max_positions_per_event)
        self.drawdown_halt_pct = float(drawdown_halt_pct)
        self.open_positions: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self.stale_halt_count = 0
        self.drawdown_halt_active = False
        self._trade_sequence = 0

    def realized_pnl(self) -> float:
        return round(sum(to_float(item.get("net_pnl_usd"), 0.0) for item in self.closed_trades), 6)

    def fees_paid(self) -> float:
        return round(sum(to_float(item.get("entry_fee_usd"), 0.0) + to_float(item.get("exit_fee_usd"), 0.0) for item in self.closed_trades), 6)

    def current_balance(self, quote_map: Dict[str, Dict[str, Any]]) -> float:
        return round(self.starting_balance + self.realized_pnl() + self.unrealized_pnl(quote_map), 6)

    def gross_exposure(self) -> float:
        return round(sum(to_float(item.get("notional_usd"), 0.0) for item in self.open_positions), 6)

    def unrealized_pnl(self, quote_map: Dict[str, Dict[str, Any]]) -> float:
        total = 0.0
        for position in self.open_positions:
            quote = dict(quote_map.get(str(position.get("token_id") or "")) or {})
            current_bid = clamp(to_float(quote.get("best_bid") or quote.get("midpoint"), 0.0))
            if current_bid <= 0:
                continue
            total += (current_bid - to_float(position.get("entry_price"), 0.0)) * to_float(position.get("quantity"), 0.0)
        return round(total, 6)

    def drawdown_pct(self, quote_map: Dict[str, Dict[str, Any]]) -> float:
        balance = self.current_balance(quote_map)
        if self.starting_balance <= 0:
            return 0.0
        return round(max(0.0, ((self.starting_balance - balance) / self.starting_balance) * 100.0), 6)

    def can_open(self, feature_row: Dict[str, Any], quote_map: Dict[str, Dict[str, Any]]) -> tuple[bool, str]:
        if self.drawdown_pct(quote_map) >= self.drawdown_halt_pct:
            self.drawdown_halt_active = True
            return False, "drawdown_halt"
        if len(self.open_positions) >= self.max_open_positions:
            return False, "max_open_positions"
        if bool(feature_row.get("resolved") or feature_row.get("closed")):
            return False, "market_closed"
        event_slug = str(feature_row.get("event_slug") or "")
        same_event = sum(1 for item in self.open_positions if str(item.get("event_slug") or "") == event_slug)
        if same_event >= self.max_positions_per_event:
            return False, "event_correlation_cap"
        token_id = str(feature_row.get("token_id") or "")
        if any(str(item.get("token_id") or "") == token_id for item in self.open_positions):
            return False, "position_already_open"
        best_ask = clamp(to_float(feature_row.get("best_ask"), 0.0))
        if best_ask <= 0:
            return False, "missing_best_ask"
        spread_bps = max(0.0, to_float(feature_row.get("spread_bps"), 0.0))
        if spread_bps > float(getattr(config, "POLYMARKET_QF_MAX_SPREAD_BPS", 250.0)):
            return False, "spread_too_wide"
        ask_depth = max(0.0, to_float(feature_row.get("ask_depth"), 0.0))
        min_depth = self.max_notional_per_trade * float(
            getattr(config, "POLYMARKET_QF_MIN_ASK_DEPTH_MULTIPLE", 1.25)
        )
        if ask_depth < min_depth:
            return False, "insufficient_ask_depth"
        return True, "ok"

    def open_trade(self, feature_row: Dict[str, Any], *, score_probability: float, notional_usd: float) -> Dict[str, Any]:
        self._trade_sequence += 1
        best_ask = clamp(to_float(feature_row.get("best_ask"), 0.0))
        tick_size = max(0.0001, to_float(feature_row.get("tick_size"), 0.001))
        desired_notional = max(1.0, min(float(notional_usd), self.max_notional_per_trade))
        desired_shares = desired_notional / max(best_ask, 0.01)
        fill_price, slippage_bps = _walk_levels(
            feature_row.get("asks") or [],
            desired_shares=desired_shares,
            fallback_price=best_ask,
            tick_size=tick_size,
            queue_penalty_bps=self.queue_penalty_bps,
            side="BUY",
        )
        quantity = desired_notional / max(fill_price, 0.01)
        entry_fee = desired_notional * (self.fee_bps / 10000.0)
        trade = {
            "trade_id": f"pmqf-{self._trade_sequence}",
            "symbol": f"{feature_row.get('market_slug')}:{feature_row.get('outcome')}",
            "market_name": feature_row.get("title"),
            "token_id": feature_row.get("token_id"),
            "market_slug": feature_row.get("market_slug"),
            "event_slug": feature_row.get("event_slug"),
            "side": "LONG",
            "status": "OPEN",
            "opened_at": utc_now_iso(),
            "entry_price": round(fill_price, 6),
            "quantity": round(quantity, 6),
            "notional_usd": round(desired_notional, 6),
            "entry_fee_usd": round(entry_fee, 6),
            "entry_slippage_bps": round(slippage_bps, 6),
            "score_probability": round(score_probability, 6),
            "score_edge": round(score_probability - 0.5, 6),
            "horizon_seconds": feature_row.get("target_horizon_seconds"),
            "strategy_context": {
                "sport": feature_row.get("sport"),
                "competition": feature_row.get("competition"),
                "folding_confidence": feature_row.get("folding_confidence"),
                "coherence_score": feature_row.get("coherence_score"),
            },
        }
        self.open_positions.append(trade)
        return trade

    def close_trade(self, position: Dict[str, Any], quote_row: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
        best_bid = clamp(to_float(quote_row.get("best_bid") or quote_row.get("midpoint"), 0.0))
        tick_size = max(0.0001, to_float(quote_row.get("tick_size"), 0.001))
        quantity = max(0.0, to_float(position.get("quantity"), 0.0))
        fill_price, slippage_bps = _walk_levels(
            quote_row.get("bids") or [],
            desired_shares=quantity,
            fallback_price=best_bid,
            tick_size=tick_size,
            queue_penalty_bps=self.queue_penalty_bps,
            side="SELL",
        )
        exit_notional = fill_price * quantity
        exit_fee = exit_notional * (self.fee_bps / 10000.0)
        net_pnl = (fill_price - to_float(position.get("entry_price"), 0.0)) * quantity
        net_pnl -= to_float(position.get("entry_fee_usd"), 0.0) + exit_fee
        closed = {
            **position,
            "status": "CLOSED",
            "closed_at": utc_now_iso(),
            "close_reason": reason,
            "exit_price": round(fill_price, 6),
            "exit_fee_usd": round(exit_fee, 6),
            "exit_slippage_bps": round(slippage_bps, 6),
            "gross_pnl_usd": round((fill_price - to_float(position.get("entry_price"), 0.0)) * quantity, 6),
            "net_pnl_usd": round(net_pnl, 6),
            "hold_seconds": max(
                0.0,
                (
                    (parse_ts(utc_now_iso()) or parse_ts(position.get("opened_at"))).timestamp()
                    - (parse_ts(position.get("opened_at")) or parse_ts(utc_now_iso())).timestamp()
                ),
            ),
        }
        self.open_positions = [item for item in self.open_positions if item.get("trade_id") != position.get("trade_id")]
        self.closed_trades.append(closed)
        self.closed_trades = self.closed_trades[-200:]
        return closed

    def execution_quality(self) -> Dict[str, Any]:
        closed = list(self.closed_trades)
        avg_slippage = 0.0
        if closed:
            avg_slippage = sum(
                to_float(item.get("entry_slippage_bps"), 0.0) + to_float(item.get("exit_slippage_bps"), 0.0)
                for item in closed
            ) / (2.0 * len(closed))
        return {
            "avg_modeled_slippage_bps": round(avg_slippage, 6),
            "queue_penalty_bps": self.queue_penalty_bps,
            "fee_bps": self.fee_bps,
            "stale_quote_halts": self.stale_halt_count,
            "drawdown_halt_active": self.drawdown_halt_active,
        }
