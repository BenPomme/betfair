"""
Paper executor for funding rate arbitrage using Binance testnet.
Executes real orders on testnet (spot buy + perp short) and tracks P&L.
Falls back to local simulation if testnet doesn't process funding.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import config
from funding.core import fee_calculator, hedge_calculator
from funding.core.schemas import FundingOpportunity, HedgePosition, HedgeStatus
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.data.binance_spot_client import BinanceSpotClient
from funding.execution.position_manager import PositionManager

logger = logging.getLogger(__name__)


class FundingPaperExecutor:
    """Paper executor using Binance testnet for spot+perp hedge execution."""

    def __init__(
        self,
        futures_client: Optional[BinanceFuturesClient] = None,
        spot_client: Optional[BinanceSpotClient] = None,
        position_manager: Optional[PositionManager] = None,
    ):
        self._futures = futures_client or BinanceFuturesClient(
            api_key=config.BINANCE_FUTURES_TESTNET_API_KEY,
            api_secret=config.BINANCE_FUTURES_TESTNET_API_SECRET,
            base_url=config.BINANCE_FUTURES_TESTNET_URL,
        )
        self._spot = spot_client or BinanceSpotClient(
            api_key=config.BINANCE_SPOT_TESTNET_API_KEY,
            api_secret=config.BINANCE_SPOT_TESTNET_API_SECRET,
            base_url=config.BINANCE_SPOT_TESTNET_URL,
        )
        self._positions = position_manager or PositionManager()
        self._simulated_fills_only = not (
            bool(config.BINANCE_FUTURES_TESTNET_API_KEY)
            and bool(config.BINANCE_FUTURES_TESTNET_API_SECRET)
            and bool(config.BINANCE_SPOT_TESTNET_API_KEY)
            and bool(config.BINANCE_SPOT_TESTNET_API_SECRET)
        )
        if self._simulated_fills_only:
            logger.warning(
                "FundingPaperExecutor running in local-sim mode (testnet API keys missing)."
            )

    @staticmethod
    def _execution_uses_maker_fees() -> bool:
        """Current hedge executor submits market orders for both legs."""
        return False

    def _estimate_entry_fees(self, notional: Decimal) -> Decimal:
        return fee_calculator.trading_fees_round_trip(
            notional,
            maker=self._execution_uses_maker_fees(),
            bnb_discount=config.FUNDING_BNB_DISCOUNT,
        ) / Decimal("2")

    def _estimate_exit_fees(self, notional: Decimal) -> Decimal:
        return fee_calculator.trading_fees_round_trip(
            notional,
            maker=self._execution_uses_maker_fees(),
            bnb_discount=config.FUNDING_BNB_DISCOUNT,
        ) / Decimal("2")

    async def open_hedge(
        self,
        opportunity: FundingOpportunity,
        exchange_filters: Optional[dict] = None,
    ) -> Optional[HedgePosition]:
        """Open a delta-neutral hedge: spot buy + perp short.

        Args:
            opportunity: The funding opportunity to execute.
            exchange_filters: Exchange filters for lot sizing.

        Returns:
            HedgePosition if successful, None if failed.
        """
        symbol = opportunity.symbol
        logger.info("Opening hedge for %s ($%s)", symbol, opportunity.position_size)

        # Calculate quantities
        spot_qty, perp_qty = hedge_calculator.calculate_quantities(
            opportunity, exchange_filters
        )
        if spot_qty <= Decimal("0"):
            logger.error("Cannot calculate quantities for %s", symbol)
            return None

        if self._simulated_fills_only:
            return self._open_hedge_simulated(opportunity, spot_qty, perp_qty)

        try:
            # Step 1: Set leverage
            await self._futures.set_leverage(symbol, config.FUNDING_LEVERAGE)

            # Step 2: Set margin type
            await self._futures.set_margin_type(symbol, config.FUNDING_MARGIN_TYPE)

            # Step 3: Place perp short FIRST (more critical leg)
            perp_order = await self._futures.place_order(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                quantity=perp_qty,
            )
            perp_fill_price = Decimal(str(perp_order.get("avgPrice", opportunity.entry_price_perp)))
            logger.info("Perp short filled: %s @ %s", symbol, perp_fill_price)

            # Step 4: Place spot buy
            try:
                spot_order = await self._spot.place_order(
                    symbol=symbol,
                    side="BUY",
                    order_type="MARKET",
                    quantity=spot_qty,
                )
                spot_fill_price = Decimal(str(spot_order.get("price", opportunity.entry_price_spot)))
                # For market orders, price may be 0 — use cummulativeQuoteQty / executedQty
                exec_qty = Decimal(str(spot_order.get("executedQty", "0")))
                cumm_quote = Decimal(str(spot_order.get("cummulativeQuoteQty", "0")))
                if exec_qty > Decimal("0") and cumm_quote > Decimal("0"):
                    spot_fill_price = (cumm_quote / exec_qty).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                logger.info("Spot buy filled: %s @ %s", symbol, spot_fill_price)
            except Exception as e:
                # Spot failed — unwind perp
                logger.error("Spot buy failed for %s: %s. Unwinding perp.", symbol, e)
                try:
                    await self._futures.place_order(
                        symbol=symbol, side="BUY", order_type="MARKET", quantity=perp_qty
                    )
                except Exception as ue:
                    logger.error("Failed to unwind perp for %s: %s", symbol, ue)
                return None

            # Calculate trading fees
            notional = spot_qty * spot_fill_price
            fees = self._estimate_entry_fees(notional)

            # Create position
            position = HedgePosition(
                symbol=symbol,
                entry_price_spot=spot_fill_price,
                entry_price_perp=perp_fill_price,
                quantity_spot=spot_qty,
                quantity_perp=perp_qty,
                leverage=config.FUNDING_LEVERAGE,
                margin_type=config.FUNDING_MARGIN_TYPE,
                entry_time=datetime.now(timezone.utc),
                trading_fees_paid=fees,
                status=HedgeStatus.OPEN,
            )
            self._positions.add_position(position)
            return position

        except Exception as e:
            if self._should_fallback_to_sim(e):
                logger.warning("Falling back to simulated hedge fill for %s due to API auth/permission error", symbol)
                return self._open_hedge_simulated(opportunity, spot_qty, perp_qty)
            logger.error("Failed to open hedge for %s: %s", symbol, e)
            return None

    async def close_hedge(self, symbol: str) -> Optional[HedgePosition]:
        """Close a hedge: close perp + sell spot.

        Returns:
            Updated HedgePosition if successful, None if failed.
        """
        position = self._positions.get_position(symbol)
        if position is None or position.status != HedgeStatus.OPEN:
            logger.warning("No open position for %s", symbol)
            return None

        logger.info("Closing hedge for %s", symbol)
        position.status = HedgeStatus.CLOSING

        if self._simulated_fills_only:
            return self._close_hedge_simulated(position)

        try:
            # Step 1: Close perp (buy to close short)
            perp_order = await self._futures.place_order(
                symbol=symbol,
                side="BUY",
                order_type="MARKET",
                quantity=position.quantity_perp,
            )
            perp_exit_price = Decimal(str(perp_order.get("avgPrice", "0")))

            # Step 2: Sell spot
            spot_order = await self._spot.place_order(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                quantity=position.quantity_spot,
            )
            exec_qty = Decimal(str(spot_order.get("executedQty", "0")))
            cumm_quote = Decimal(str(spot_order.get("cummulativeQuoteQty", "0")))
            if exec_qty > Decimal("0") and cumm_quote > Decimal("0"):
                spot_exit_price = (cumm_quote / exec_qty).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            else:
                spot_exit_price = Decimal(str(spot_order.get("price", "0")))

            # Calculate P&L from price changes
            spot_pnl = (spot_exit_price - position.entry_price_spot) * position.quantity_spot
            perp_pnl = (position.entry_price_perp - perp_exit_price) * position.quantity_perp
            price_pnl = spot_pnl + perp_pnl

            # Exit trading fees
            notional = position.quantity_spot * spot_exit_price
            exit_fees = self._estimate_exit_fees(notional)

            self._positions.close_position(
                symbol=symbol,
                exit_price_spot=spot_exit_price,
                exit_price_perp=perp_exit_price,
                exit_pnl=price_pnl,
                trading_fees=exit_fees,
            )

            return self._positions.get_position(symbol)

        except Exception as e:
            if self._should_fallback_to_sim(e):
                logger.warning("Falling back to simulated hedge close for %s due to API auth/permission error", symbol)
                return self._close_hedge_simulated(position)
            logger.error("Failed to close hedge for %s: %s", symbol, e)
            position.status = HedgeStatus.OPEN  # Revert to open
            return None

    @property
    def position_manager(self) -> PositionManager:
        return self._positions

    def _open_hedge_simulated(
        self,
        opportunity: FundingOpportunity,
        spot_qty: Decimal,
        perp_qty: Decimal,
    ) -> HedgePosition:
        spot_fill_price = opportunity.entry_price_spot
        perp_fill_price = opportunity.entry_price_perp
        notional = spot_qty * spot_fill_price
        fees = self._estimate_entry_fees(notional)
        position = HedgePosition(
            symbol=opportunity.symbol,
            entry_price_spot=spot_fill_price,
            entry_price_perp=perp_fill_price,
            quantity_spot=spot_qty,
            quantity_perp=perp_qty,
            leverage=config.FUNDING_LEVERAGE,
            margin_type=config.FUNDING_MARGIN_TYPE,
            entry_time=datetime.now(timezone.utc),
            trading_fees_paid=fees,
            status=HedgeStatus.OPEN,
        )
        self._positions.add_position(position)
        return position

    def _close_hedge_simulated(self, position: HedgePosition) -> HedgePosition:
        spot_exit_price = position.entry_price_spot
        perp_exit_price = position.entry_price_perp
        price_pnl = Decimal("0")
        notional = position.quantity_spot * spot_exit_price
        exit_fees = self._estimate_exit_fees(notional)
        closed = self._positions.close_position(
            symbol=position.symbol,
            exit_price_spot=spot_exit_price,
            exit_price_perp=perp_exit_price,
            exit_pnl=price_pnl,
            trading_fees=exit_fees,
        )
        return closed if closed is not None else position

    @staticmethod
    def _should_fallback_to_sim(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("-2015" in msg) or ("invalid api-key" in msg) or ("permissions for action" in msg)
