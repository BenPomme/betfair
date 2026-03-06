from __future__ import annotations

from datetime import datetime, timezone

from funding.portfolios.cascade_alpha.beta_hedge import suggest_beta_hedge


class CascadePaperExecutor:
    def __init__(self) -> None:
        self._open_positions: list[dict] = []
        self._closed_trades: list[dict] = []
        self._trade_id = 0

    @property
    def open_positions(self) -> list[dict]:
        return list(self._open_positions)

    @property
    def closed_trades(self) -> list[dict]:
        return list(self._closed_trades)

    def _fill_price(self, reference_price: float, side: str, spread_bps: float, extra_slippage_bps: float) -> float:
        total_bps = max(0.0, spread_bps / 2.0 + extra_slippage_bps)
        direction = 1.0 if side.upper() == "LONG" else -1.0
        return reference_price * (1.0 + (direction * total_bps / 10000.0))

    def open_trade(self, signal: dict) -> dict:
        self._trade_id += 1
        hedge_meta = suggest_beta_hedge(signal["symbol"])
        reference_price = float(signal["reference_price"])
        spread_bps = float(signal.get("spread_bps", 0.0))
        slippage_bps = float(signal.get("slippage_bps", 0.0))
        entry_price = self._fill_price(reference_price, signal["side"], spread_bps, slippage_bps)
        notional = float(signal["notional_usd"])
        qty = notional / entry_price if entry_price > 0 else 0.0
        entry_fee = notional * 8.0 / 10000.0
        position = {
            "trade_id": f"cascade-{self._trade_id}",
            "symbol": signal["symbol"],
            "setup": signal["setup"],
            "side": signal["side"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "reference_price": reference_price,
            "entry_price": entry_price,
            "quantity": qty,
            "notional_usd": notional,
            "entry_fee_usd": entry_fee,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "signal_score": signal.get("signal_score", 0.0),
            "taker_imbalance": signal.get("taker_imbalance", 0.0),
            "beta_hedge_symbol": hedge_meta["hedge_symbol"],
            "beta_hedge_ratio": hedge_meta["hedge_ratio"],
            "status": "OPEN",
        }
        self._open_positions.append(position)
        return position

    def close_position(self, position: dict, current_price: float, reason: str) -> dict:
        spread_bps = float(position.get("spread_bps", 0.0) or 0.0)
        slippage_bps = float(position.get("slippage_bps", 0.0) or 0.0)
        exit_side = "SHORT" if str(position.get("side", "LONG")).upper() == "LONG" else "LONG"
        exit_price = self._fill_price(current_price, exit_side, spread_bps, slippage_bps)
        side_mult = 1.0 if str(position.get("side", "LONG")).upper() == "LONG" else -1.0
        gross_pnl = (exit_price - float(position["entry_price"])) * float(position["quantity"]) * side_mult
        notional = float(position["notional_usd"])
        exit_fee = notional * 8.0 / 10000.0
        trade = dict(position)
        trade.update(
            {
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "exit_price": exit_price,
                "exit_fee_usd": exit_fee,
                "gross_pnl_usd": gross_pnl,
                "net_pnl_usd": gross_pnl - float(position.get("entry_fee_usd", 0.0)) - exit_fee,
                "status": "CLOSED",
                "close_reason": reason,
            }
        )
        self._open_positions = [item for item in self._open_positions if item.get("trade_id") != position.get("trade_id")]
        self._closed_trades.append(trade)
        self._closed_trades = self._closed_trades[-500:]
        return trade
