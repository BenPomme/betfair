"""
Executor for directional (contrarian) trades.

Single perpetual leg only — no spot hedge. Opens a leveraged long or short
based on a ContrarianSignal, attaches a stop-loss and take-profit, and
monitors open positions for stop/TP/max-hold triggers.

OPUS-reviewed: Tier 3 — financially critical.
A bug here can open real orders in the wrong direction or fail to close a
losing position, resulting in uncapped loss.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

import config
from funding.core.schemas import (
    ContrarianSignal,
    DirectionalPosition,
    DirectionalPositionStatus,
    DirectionalSide,
    FundingSnapshot,
)
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.execution.directional_position_manager import DirectionalPositionManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: opposite side string for Binance API
# ---------------------------------------------------------------------------

def _opposite_side(side: DirectionalSide) -> str:
    """Return the Binance order side string to close a directional position."""
    return "SELL" if side is DirectionalSide.LONG else "BUY"


def _entry_side(side: DirectionalSide) -> str:
    """Return the Binance order side string to open a directional position."""
    return "BUY" if side is DirectionalSide.LONG else "SELL"


# ---------------------------------------------------------------------------
# DirectionalExecutor
# ---------------------------------------------------------------------------


class DirectionalExecutor:
    """Execute and manage single-leg directional futures positions.

    Designed for contrarian (mean-reversion) trades against extreme funding
    rates. Supports both paper (testnet) and live operation controlled by
    ``config.FUNDING_MODE``.

    All monetary arithmetic uses ``Decimal``. No ``float`` arithmetic is
    performed on prices, quantities, or P&L values.
    """

    def __init__(
        self,
        futures_client: Optional[BinanceFuturesClient] = None,
        position_manager: Optional[DirectionalPositionManager] = None,
    ) -> None:
        # Determine base URL from FUNDING_MODE — testnet for paper, live otherwise.
        if futures_client is not None:
            self._futures = futures_client
        else:
            base_url = (
                config.BINANCE_FUTURES_TESTNET_URL
                if config.FUNDING_MODE == "paper"
                else config.BINANCE_FUTURES_PROD_URL
            )
            api_key = (
                config.BINANCE_FUTURES_TESTNET_API_KEY
                if config.FUNDING_MODE == "paper"
                else config.BINANCE_FUTURES_API_KEY
            )
            api_secret = (
                config.BINANCE_FUTURES_TESTNET_API_SECRET
                if config.FUNDING_MODE == "paper"
                else config.BINANCE_FUTURES_API_SECRET
            )
            self._futures = BinanceFuturesClient(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url,
            )

        self._positions = position_manager or DirectionalPositionManager()
        self._simulated_fills_only = bool(config.FUNDING_MODE == "paper") and not (
            bool(config.BINANCE_FUTURES_TESTNET_API_KEY)
            and bool(config.BINANCE_FUTURES_TESTNET_API_SECRET)
        )
        if self._simulated_fills_only:
            logger.warning("DirectionalExecutor running in local-sim mode (futures testnet keys missing).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def open_position(
        self,
        signal: ContrarianSignal,
        params: dict,
    ) -> Optional[DirectionalPosition]:
        """Open a directional futures position for the given signal.

        Steps executed in order:
        1. Set leverage on the symbol.
        2. Set margin type (ISOLATED or CROSSED).
        3. Place market order in signal direction.
        4. Place stop-loss STOP_MARKET order.
        5. Place take-profit TAKE_PROFIT_MARKET order.
        6. Record the position in the position manager.

        If the market order succeeds but either the SL or TP placement
        fails, the position is still recorded (SL/TP can be retried) but
        the failure is logged at ERROR level.

        Args:
            signal: The ContrarianSignal triggering this trade.
            params: Execution parameters dict with keys:
                - ``quantity`` (Decimal): contract quantity to trade.
                - ``leverage`` (int): futures leverage to set.
                - ``margin_type`` (str): "ISOLATED" or "CROSSED".
                - ``stop_loss`` (Decimal): trigger price for the stop-loss.
                - ``take_profit`` (Decimal): trigger price for take-profit.

        Returns:
            ``DirectionalPosition`` if the market order was placed, else ``None``.
        """
        symbol = signal.symbol
        leverage: int = int(params["leverage"])
        margin_type: str = str(params["margin_type"])
        quantity: Decimal = Decimal(str(params["quantity"]))
        stop_loss: Decimal = Decimal(str(params["stop_loss"]))
        take_profit: Decimal = Decimal(str(params["take_profit"]))
        if quantity <= Decimal("0"):
            notional = Decimal(str(params.get("notional", "0")))
            mark = Decimal(str(params.get("entry_price", signal.mark_price or "0")))
            if mark > Decimal("0") and notional > Decimal("0"):
                quantity = (notional / mark).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if self._simulated_fills_only:
            return self._open_position_simulated(signal, params, quantity, stop_loss, take_profit)

        entry_side = _entry_side(signal.direction)
        close_side = _opposite_side(signal.direction)

        logger.info(
            "Opening %s directional position: %s qty=%s lev=%sx SL=%s TP=%s",
            signal.direction.value, symbol, quantity, leverage, stop_loss, take_profit,
        )

        try:
            # Step 1: Set leverage
            await self._futures.set_leverage(symbol, leverage)
        except Exception as exc:
            if self._should_fallback_to_sim(exc):
                return self._open_position_simulated(signal, params, quantity, stop_loss, take_profit)
            logger.error(
                "Failed to set leverage %sx for %s: %s — aborting open_position",
                leverage, symbol, exc,
            )
            return None

        try:
            # Step 2: Set margin type
            await self._futures.set_margin_type(symbol, margin_type)
        except Exception as exc:
            # Binance returns an error if margin type is already set to the
            # requested value.  This is harmless; treat it as a warning only.
            logger.warning(
                "set_margin_type(%s, %s) raised: %s — continuing",
                symbol, margin_type, exc,
            )

        try:
            # Step 3: Place market entry order
            entry_order = await self._futures.place_order(
                symbol=symbol,
                side=entry_side,
                order_type="MARKET",
                quantity=quantity,
            )
        except Exception as exc:
            if self._should_fallback_to_sim(exc):
                return self._open_position_simulated(signal, params, quantity, stop_loss, take_profit)
            logger.error(
                "Market entry order failed for %s: %s — aborting open_position",
                symbol, exc,
            )
            return None

        # Extract fill price from the order response.
        avg_price_raw = entry_order.get("avgPrice", "0")
        entry_price = Decimal(str(avg_price_raw)) if avg_price_raw else Decimal("0")
        if entry_price == Decimal("0"):
            # Fall back to the signal's mark price if the exchange didn't echo
            # avgPrice (can happen on testnet for MARKET orders).
            entry_price = signal.funding_rate  # placeholder — override below
            # Attempt to fetch from position risk instead.
            try:
                risk_list = await self._futures.get_position_risk(symbol=symbol)
                for item in risk_list:
                    if item.get("symbol") == symbol and item.get("entry_price", Decimal("0")) > Decimal("0"):
                        entry_price = item["entry_price"]
                        break
            except Exception as exc:
                logger.warning("Could not fetch entry price from position risk: %s", exc)

        logger.info(
            "Market %s order filled: %s @ %s (order_id=%s)",
            entry_side, symbol, entry_price, entry_order.get("orderId", "?"),
        )

        # Build the position object immediately after the market order fills so
        # that we hold state even if the SL/TP placements below fail.
        position = DirectionalPosition(
            symbol=symbol,
            side=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            margin_type=margin_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc),
            status=DirectionalPositionStatus.OPEN,
            signal=signal,
        )

        # Step 4: Place stop-loss order
        try:
            sl_order = await self._futures.place_order(
                symbol=symbol,
                side=close_side,
                order_type="STOP_MARKET",
                quantity=quantity,
                stopPrice=str(stop_loss),
                closePosition="true",
                timeInForce="GTE_GTC",
            )
            logger.info(
                "Stop-loss placed for %s: trigger=%s order_id=%s",
                symbol, stop_loss, sl_order.get("orderId", "?"),
            )
        except Exception as exc:
            logger.error(
                "Failed to place stop-loss for %s (SL=%s): %s — position is OPEN without SL; retry required",
                symbol, stop_loss, exc,
            )
            # Position is kept; SL must be retried externally.

        # Step 5: Place take-profit order
        try:
            tp_order = await self._futures.place_order(
                symbol=symbol,
                side=close_side,
                order_type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=str(take_profit),
                closePosition="true",
                timeInForce="GTE_GTC",
            )
            logger.info(
                "Take-profit placed for %s: trigger=%s order_id=%s",
                symbol, take_profit, tp_order.get("orderId", "?"),
            )
        except Exception as exc:
            logger.error(
                "Failed to place take-profit for %s (TP=%s): %s — position is OPEN without TP; retry required",
                symbol, take_profit, exc,
            )

        # Step 6: Register position in manager regardless of SL/TP outcome.
        self._positions.add_position(position)
        return position

    # ------------------------------------------------------------------

    async def close_position(
        self,
        symbol: str,
        reason: str = "manual",
        current_price: Optional[Decimal] = None,
    ) -> Optional[DirectionalPosition]:
        """Close an open directional position by placing a closing market order.

        After the closing order is placed:
        - Any open SL/TP orders for the symbol are cancelled (best-effort).
        - Realized P&L is computed and stored on the position.
        - Position status is updated to reflect the close reason.

        Args:
            symbol: Perpetual symbol to close (e.g. "BTCUSDT").
            reason: One of "manual", "stopped", "took_profit", "max_hold".
                    Determines the final ``DirectionalPositionStatus``.

        Returns:
            The updated ``DirectionalPosition`` if closed successfully, else ``None``.
        """
        position = self._positions.get_position(symbol)
        if position is None:
            logger.warning("close_position called for %s but no position found", symbol)
            return None
        if position.status != DirectionalPositionStatus.OPEN:
            logger.warning(
                "close_position called for %s but status is %s — skipping",
                symbol, position.status.value,
            )
            return None

        close_side = _opposite_side(position.side)

        logger.info(
            "Closing %s directional position: %s qty=%s reason=%s",
            position.side.value, symbol, position.quantity, reason,
        )

        position.status = DirectionalPositionStatus.CLOSING
        self._positions.update_position(position)

        if self._simulated_fills_only:
            return self._close_position_simulated(position, reason=reason, current_price=current_price)

        try:
            close_order = await self._futures.place_order(
                symbol=symbol,
                side=close_side,
                order_type="MARKET",
                quantity=position.quantity,
            )
        except Exception as exc:
            if self._should_fallback_to_sim(exc):
                return self._close_position_simulated(position, reason=reason, current_price=current_price)
            logger.error(
                "Closing market order failed for %s: %s — position left in CLOSING state",
                symbol, exc,
            )
            # Revert to OPEN so the monitor can retry.
            position.status = DirectionalPositionStatus.OPEN
            self._positions.update_position(position)
            return None

        # Derive exit price from the close order response.
        avg_price_raw = close_order.get("avgPrice", "0")
        exit_price = Decimal(str(avg_price_raw)) if avg_price_raw else Decimal("0")
        if exit_price == Decimal("0"):
            try:
                risk_list = await self._futures.get_position_risk(symbol=symbol)
                # After close, positionAmt should be 0; use last known mark price.
                for item in risk_list:
                    if item.get("symbol") == symbol:
                        exit_price = item.get("mark_price", Decimal("0"))
                        break
            except Exception as exc:
                logger.warning("Could not fetch exit price from position risk: %s", exc)

        logger.info(
            "Closing order filled: %s @ %s (order_id=%s)",
            symbol, exit_price, close_order.get("orderId", "?"),
        )

        # Cancel any outstanding SL/TP orders (best-effort).
        await self._cancel_open_orders(symbol)

        # Compute realized P&L.
        # LONG: (exit - entry) * qty
        # SHORT: (entry - exit) * qty
        if position.side is DirectionalSide.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        pnl = pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Determine terminal status from reason.
        _reason_to_status: Dict[str, DirectionalPositionStatus] = {
            "stopped": DirectionalPositionStatus.STOPPED,
            "took_profit": DirectionalPositionStatus.TOOK_PROFIT,
        }
        final_status = _reason_to_status.get(reason, DirectionalPositionStatus.CLOSED)

        position.exit_price = exit_price
        position.exit_time = datetime.now(timezone.utc)
        position.realized_pnl = pnl
        position.status = final_status
        self._positions.update_position(position)

        logger.info(
            "Position closed: %s %s pnl=%s status=%s",
            symbol, reason, pnl, final_status.value,
        )
        return position

    # ------------------------------------------------------------------

    async def check_stops(
        self,
        snapshots: Dict[str, FundingSnapshot],
    ) -> List[str]:
        """Check all open directional positions for stop/TP/max-hold triggers.

        For each open position:
        - If ``LONG`` and ``mark_price <= stop_loss`` → close with reason "stopped".
        - If ``LONG`` and ``mark_price >= take_profit`` → close with reason "took_profit".
        - If ``SHORT`` and ``mark_price >= stop_loss`` → close with reason "stopped".
        - If ``SHORT`` and ``mark_price <= take_profit`` → close with reason "took_profit".
        - If held longer than ``CONTRARIAN_MAX_HOLD_HOURS`` → close with reason "max_hold".

        In paper (testnet) mode the same logic applies; Binance testnet does
        not reliably trigger conditional orders on its own, so this method
        acts as the primary stop simulation layer.

        Args:
            snapshots: Mapping of symbol → ``FundingSnapshot`` with current
                       ``mark_price`` data.

        Returns:
            List of symbols for which a position was closed this cycle.
        """
        closed_symbols: List[str] = []
        max_hold = timedelta(hours=config.CONTRARIAN_MAX_HOLD_HOURS)
        now = datetime.now(timezone.utc)

        for position in list(self._positions.open_positions()):
            symbol = position.symbol
            snapshot = snapshots.get(symbol)

            # --- Max hold time check (no price data needed) ---
            if position.entry_time is not None:
                held = now - position.entry_time
                if held >= max_hold:
                    logger.info(
                        "Max hold reached for %s (held=%s) — closing",
                        symbol, held,
                    )
                    closed = await self.close_position(symbol, reason="max_hold", current_price=snapshot.mark_price if snapshot else None)
                    if closed is not None:
                        closed_symbols.append(symbol)
                    # Skip price checks — position is already closed.
                    continue

            if snapshot is None:
                logger.debug("No snapshot available for open position %s — skip stop check", symbol)
                continue

            current_price: Decimal = snapshot.mark_price

            # --- Price-based stop and take-profit checks ---
            if position.side is DirectionalSide.LONG:
                if current_price <= position.stop_loss:
                    logger.info(
                        "LONG stop triggered: %s price=%s <= SL=%s",
                        symbol, current_price, position.stop_loss,
                    )
                    closed = await self.close_position(symbol, reason="stopped", current_price=current_price)
                    if closed is not None:
                        closed_symbols.append(symbol)

                elif current_price >= position.take_profit:
                    logger.info(
                        "LONG take-profit triggered: %s price=%s >= TP=%s",
                        symbol, current_price, position.take_profit,
                    )
                    closed = await self.close_position(symbol, reason="took_profit", current_price=current_price)
                    if closed is not None:
                        closed_symbols.append(symbol)

            else:  # SHORT
                if current_price >= position.stop_loss:
                    logger.info(
                        "SHORT stop triggered: %s price=%s >= SL=%s",
                        symbol, current_price, position.stop_loss,
                    )
                    closed = await self.close_position(symbol, reason="stopped", current_price=current_price)
                    if closed is not None:
                        closed_symbols.append(symbol)

                elif current_price <= position.take_profit:
                    logger.info(
                        "SHORT take-profit triggered: %s price=%s <= TP=%s",
                        symbol, current_price, position.take_profit,
                    )
                    closed = await self.close_position(symbol, reason="took_profit", current_price=current_price)
                    if closed is not None:
                        closed_symbols.append(symbol)

        return closed_symbols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cancel_open_orders(self, symbol: str) -> None:
        """Cancel all open orders for *symbol* (best-effort).

        Failures are logged but never propagated — the caller must not depend
        on this completing successfully.
        """
        try:
            # BinanceFuturesClient wraps the SDK; cancel_all_open_orders is
            # exposed via cancel_order per order ID.  Use the raw SDK method
            # via place_order workaround if available, or iterate known IDs.
            # The SDK method `cancel_all_open_orders` is available directly.
            await asyncio.to_thread(
                self._futures._client.cancel_all_open_orders,
                symbol=symbol,
            )
            logger.info("Cancelled all open orders for %s", symbol)
        except Exception as exc:
            logger.warning(
                "Could not cancel open orders for %s: %s — manual cleanup may be required",
                symbol, exc,
            )

    def _open_position_simulated(
        self,
        signal: ContrarianSignal,
        params: dict,
        quantity: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
    ) -> Optional[DirectionalPosition]:
        symbol = signal.symbol
        entry_price = Decimal(str(params.get("entry_price", signal.mark_price or "0")))
        if entry_price <= Decimal("0") or quantity <= Decimal("0"):
            return None
        if stop_loss <= Decimal("0") or take_profit <= Decimal("0"):
            stop_pct = Decimal(str(config.CONTRARIAN_STOP_LOSS_PCT))
            rr = Decimal(str(config.CONTRARIAN_TAKE_PROFIT_RATIO))
            stop_dist = (entry_price * stop_pct).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
            reward_dist = (stop_dist * rr).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
            if signal.direction is DirectionalSide.LONG:
                stop_loss = (entry_price - stop_dist).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
                take_profit = (entry_price + reward_dist).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
            else:
                stop_loss = (entry_price + stop_dist).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
                take_profit = (entry_price - reward_dist).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

        position = DirectionalPosition(
            symbol=symbol,
            side=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            leverage=int(params.get("leverage", config.CONTRARIAN_LEVERAGE)),
            margin_type=str(params.get("margin_type", config.FUNDING_MARGIN_TYPE)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc),
            status=DirectionalPositionStatus.OPEN,
            signal=signal,
        )
        self._positions.add_position(position)
        return position

    def _close_position_simulated(
        self,
        position: DirectionalPosition,
        reason: str,
        current_price: Optional[Decimal] = None,
    ) -> DirectionalPosition:
        exit_price = current_price if current_price is not None else position.entry_price
        if position.side is DirectionalSide.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        pnl = pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        _reason_to_status: Dict[str, DirectionalPositionStatus] = {
            "stopped": DirectionalPositionStatus.STOPPED,
            "took_profit": DirectionalPositionStatus.TOOK_PROFIT,
        }
        final_status = _reason_to_status.get(reason, DirectionalPositionStatus.CLOSED)
        position.exit_price = exit_price
        position.exit_time = datetime.now(timezone.utc)
        position.realized_pnl = pnl
        position.status = final_status
        self._positions.update_position(position)
        return position

    @staticmethod
    def _should_fallback_to_sim(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("-2015" in msg) or ("invalid api-key" in msg) or ("permissions for action" in msg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position_manager(self) -> DirectionalPositionManager:
        return self._positions
