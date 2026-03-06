"""
Paper executor for funding rate arbitrage using Binance testnet.
Executes real orders on testnet (spot buy + perp short) and tracks P&L.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple

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
        self._validation_mode = bool(config.FUNDING_VALIDATION_MODE)
        self._require_testnet_fills = bool(config.FUNDING_PAPER_REQUIRE_TESTNET_FILLS)
        self._allow_sim_fallback = bool(config.FUNDING_PAPER_ALLOW_SIM_FALLBACK)
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

    @staticmethod
    def _round_bps(value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

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

    def _split_entry_fees(self, notional: Decimal) -> Tuple[Decimal, Decimal]:
        return (
            fee_calculator.spot_fee(notional, maker=False, bnb_discount=config.FUNDING_BNB_DISCOUNT),
            fee_calculator.futures_fee(notional, maker=False, bnb_discount=config.FUNDING_BNB_DISCOUNT),
        )

    def _split_exit_fees(self, notional: Decimal) -> Tuple[Decimal, Decimal]:
        return (
            fee_calculator.spot_fee(notional, maker=False, bnb_discount=config.FUNDING_BNB_DISCOUNT),
            fee_calculator.futures_fee(notional, maker=False, bnb_discount=config.FUNDING_BNB_DISCOUNT),
        )

    def _reject(self, reason: str, symbol: str, details: Optional[dict] = None) -> None:
        self._positions.log_rejection(reason, symbol, details=details)

    @staticmethod
    def _order_id(order: dict) -> Optional[str]:
        order_id = order.get("orderId") or order.get("order_id") or order.get("clientOrderId")
        return str(order_id) if order_id is not None else None

    def _extract_futures_fill_price(self, order: dict) -> Optional[Decimal]:
        avg_price = Decimal(str(order.get("avgPrice", "0")))
        if avg_price > Decimal("0"):
            return avg_price
        executed_qty = Decimal(str(order.get("executedQty", "0")))
        cum_quote = Decimal(str(order.get("cumQuote", order.get("cumQuoteQty", "0"))))
        if executed_qty > Decimal("0") and cum_quote > Decimal("0"):
            return (cum_quote / executed_qty).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return None

    def _extract_spot_fill_price(self, order: dict) -> Optional[Decimal]:
        executed_qty = Decimal(str(order.get("executedQty", "0")))
        cumulative_quote = Decimal(str(order.get("cummulativeQuoteQty", "0")))
        if executed_qty > Decimal("0") and cumulative_quote > Decimal("0"):
            return (cumulative_quote / executed_qty).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        price = Decimal(str(order.get("price", "0")))
        return price if price > Decimal("0") else None

    def _slippage_bps(self, fill_price: Decimal, reference_price: Optional[Decimal], side: str) -> Decimal:
        if reference_price is None or reference_price <= Decimal("0") or fill_price <= Decimal("0"):
            return Decimal("0")
        if side.upper() == "BUY":
            return self._round_bps(((fill_price - reference_price) / reference_price) * Decimal("10000"))
        return self._round_bps(((reference_price - fill_price) / reference_price) * Decimal("10000"))

    async def open_hedge(
        self,
        opportunity: FundingOpportunity,
        exchange_filters: Optional[dict] = None,
    ) -> Optional[HedgePosition]:
        """Open a delta-neutral hedge: spot buy + perp short."""
        symbol = opportunity.symbol
        logger.info("Opening hedge for %s ($%s)", symbol, opportunity.position_size)

        if opportunity.rejection_reason:
            self._reject(opportunity.rejection_reason, symbol, details={"stage": "precheck"})
            return None

        spot_qty, perp_qty = hedge_calculator.calculate_quantities(opportunity, exchange_filters)
        if spot_qty <= Decimal("0"):
            logger.error("Cannot calculate quantities for %s", symbol)
            self._reject("quantity_invalid", symbol, details={"position_size": str(opportunity.position_size)})
            return None

        if self._simulated_fills_only:
            if self._validation_mode or self._require_testnet_fills or not self._allow_sim_fallback:
                self._reject("missing_testnet_credentials", symbol, details={"stage": "entry"})
                return None
            return self._open_hedge_simulated(opportunity, spot_qty, perp_qty)

        try:
            await self._futures.set_leverage(symbol, config.FUNDING_LEVERAGE)
            await self._futures.set_margin_type(symbol, config.FUNDING_MARGIN_TYPE)

            perp_order = await self._futures.place_order(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                quantity=perp_qty,
            )
            perp_fill_price = self._extract_futures_fill_price(perp_order)
            if perp_fill_price is None:
                self._reject("perp_order_failed", symbol, details={"order": perp_order, "stage": "entry"})
                return None
            logger.info("Perp short filled: %s @ %s", symbol, perp_fill_price)

            try:
                spot_order = await self._spot.place_order(
                    symbol=symbol,
                    side="BUY",
                    order_type="MARKET",
                    quantity=spot_qty,
                )
                spot_fill_price = self._extract_spot_fill_price(spot_order)
                if spot_fill_price is None:
                    raise RuntimeError(f"missing spot fill price: {spot_order}")
                logger.info("Spot buy filled: %s @ %s", symbol, spot_fill_price)
            except Exception as exc:
                logger.error("Spot buy failed for %s: %s. Unwinding perp.", symbol, exc)
                try:
                    await self._futures.place_order(
                        symbol=symbol,
                        side="BUY",
                        order_type="MARKET",
                        quantity=perp_qty,
                    )
                except Exception as unwind_exc:
                    self._reject("unwind_failed", symbol, details={"stage": "entry", "error": str(unwind_exc)})
                    logger.error("Failed to unwind perp for %s: %s", symbol, unwind_exc)
                    return None
                self._reject("spot_order_failed", symbol, details={"stage": "entry", "error": str(exc)})
                return None

            notional = spot_qty * spot_fill_price
            entry_fee_spot, entry_fee_perp = self._split_entry_fees(notional)
            total_fees = entry_fee_spot + entry_fee_perp
            spot_ref = opportunity.spot_ask or opportunity.entry_price_spot
            perp_ref = opportunity.perp_bid or opportunity.entry_price_perp

            position = HedgePosition(
                symbol=symbol,
                entry_price_spot=spot_fill_price,
                entry_price_perp=perp_fill_price,
                quantity_spot=spot_qty,
                quantity_perp=perp_qty,
                leverage=config.FUNDING_LEVERAGE,
                margin_type=config.FUNDING_MARGIN_TYPE,
                entry_time=datetime.now(timezone.utc),
                trading_fees_paid=total_fees,
                status=HedgeStatus.OPEN,
                validation_run_id=self._positions.get_validation_context().get("validation_run_id", ""),
                fill_source="exchange_testnet",
                entry_order_id_spot=self._order_id(spot_order),
                entry_order_id_perp=self._order_id(perp_order),
                entry_basis_bps=self._round_bps(opportunity.basis_bps),
                entry_slippage_bps_spot=self._slippage_bps(spot_fill_price, spot_ref, "BUY"),
                entry_slippage_bps_perp=self._slippage_bps(perp_fill_price, perp_ref, "SELL"),
                entry_fee_spot=entry_fee_spot,
                entry_fee_perp=entry_fee_perp,
                expected_funding_payment=opportunity.expected_funding_payment,
            )
            self._positions.add_position(position)
            return position

        except Exception as exc:
            if self._should_fallback_to_sim(exc) and not (
                self._validation_mode or self._require_testnet_fills or not self._allow_sim_fallback
            ):
                logger.warning(
                    "Falling back to simulated hedge fill for %s due to API auth/permission error",
                    symbol,
                )
                return self._open_hedge_simulated(opportunity, spot_qty, perp_qty)
            self._reject("perp_order_failed", symbol, details={"stage": "entry", "error": str(exc)})
            logger.error("Failed to open hedge for %s: %s", symbol, exc)
            return None

    async def close_hedge(self, symbol: str) -> Optional[HedgePosition]:
        """Close a hedge: close perp + sell spot."""
        position = self._positions.get_position(symbol)
        if position is None or position.status != HedgeStatus.OPEN:
            logger.warning("No open position for %s", symbol)
            return None

        logger.info("Closing hedge for %s", symbol)
        position.status = HedgeStatus.CLOSING

        if self._simulated_fills_only:
            if self._validation_mode or self._require_testnet_fills or not self._allow_sim_fallback:
                self._reject("missing_testnet_credentials", symbol, details={"stage": "close"})
                position.status = HedgeStatus.OPEN
                return None
            return self._close_hedge_simulated(position)

        try:
            spot_book = await self._spot.get_order_book(symbol, limit=5)
            perp_book = await self._futures.get_order_book(symbol, limit=5)
            spot_ref_bid = Decimal(str((spot_book.get("bids") or [[position.entry_price_spot, "0"]])[0][0]))
            perp_ref_ask = Decimal(str((perp_book.get("asks") or [[position.entry_price_perp, "0"]])[0][0]))

            perp_order = await self._futures.place_order(
                symbol=symbol,
                side="BUY",
                order_type="MARKET",
                quantity=position.quantity_perp,
            )
            perp_exit_price = self._extract_futures_fill_price(perp_order)
            if perp_exit_price is None:
                self._reject("perp_order_failed", symbol, details={"stage": "close", "order": perp_order})
                position.status = HedgeStatus.OPEN
                return None

            spot_order = await self._spot.place_order(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                quantity=position.quantity_spot,
            )
            spot_exit_price = self._extract_spot_fill_price(spot_order)
            if spot_exit_price is None:
                self._reject("spot_order_failed", symbol, details={"stage": "close", "order": spot_order})
                position.status = HedgeStatus.OPEN
                return None

            spot_pnl = (spot_exit_price - position.entry_price_spot) * position.quantity_spot
            perp_pnl = (position.entry_price_perp - perp_exit_price) * position.quantity_perp
            price_pnl = spot_pnl + perp_pnl

            notional = position.quantity_spot * spot_exit_price
            exit_fee_spot, exit_fee_perp = self._split_exit_fees(notional)
            exit_fees = exit_fee_spot + exit_fee_perp

            self._positions.close_position(
                symbol=symbol,
                exit_price_spot=spot_exit_price,
                exit_price_perp=perp_exit_price,
                exit_pnl=price_pnl,
                trading_fees=exit_fees,
            )
            exit_basis_bps = Decimal("0")
            if spot_ref_bid > Decimal("0"):
                exit_basis_bps = self._round_bps(((perp_ref_ask - spot_ref_bid) / spot_ref_bid) * Decimal("10000"))

            closed = self._positions.update_position(
                symbol,
                fill_source="exchange_testnet",
                exit_order_id_spot=self._order_id(spot_order),
                exit_order_id_perp=self._order_id(perp_order),
                exit_basis_bps=exit_basis_bps,
                exit_slippage_bps_spot=self._slippage_bps(spot_exit_price, spot_ref_bid, "SELL"),
                exit_slippage_bps_perp=self._slippage_bps(perp_exit_price, perp_ref_ask, "BUY"),
                exit_fee_spot=exit_fee_spot,
                exit_fee_perp=exit_fee_perp,
            )
            return closed

        except Exception as exc:
            if self._should_fallback_to_sim(exc) and not (
                self._validation_mode or self._require_testnet_fills or not self._allow_sim_fallback
            ):
                logger.warning(
                    "Falling back to simulated hedge close for %s due to API auth/permission error",
                    symbol,
                )
                return self._close_hedge_simulated(position)
            self._reject("perp_order_failed", symbol, details={"stage": "close", "error": str(exc)})
            logger.error("Failed to close hedge for %s: %s", symbol, exc)
            position.status = HedgeStatus.OPEN
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
            fill_source="simulated",
            expected_funding_payment=opportunity.expected_funding_payment,
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
