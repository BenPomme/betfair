"""
Main event loop for funding rate arbitrage.
Orchestrates: watchlist → WebSocket → scan → entry/exit → funding tracking.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config
from funding.core import opportunity_scanner, risk_manager
from funding.core.schemas import DirectionalSide, FundingOpportunity, FundingSnapshot, HedgePosition, HedgeStatus
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.data.binance_spot_client import BinanceSpotClient
from funding.data.market_data_stream import MarketDataStream
from funding.data.price_cache import FundingPriceCache
from funding.execution import executor
from funding.ml.online_learner import FundingOnlineLearner
from funding.ml.contrarian_learner import ContrarianOnlineLearner
from funding.strategy import entry_strategy, exit_strategy, symbol_selector
from funding.strategy.symbol_selector import SymbolSelector
from funding.utils.async_compat import async_timeout

# Optional data collectors — imported unconditionally; instantiated conditionally.
from funding.data.liquidation_stream import LiquidationStream
from funding.data.sentiment_collector import SentimentCollector
from funding.data.depth_collector import DepthCollector

# Optional contrarian strategy components — imported unconditionally; instantiated conditionally.
from funding.strategy.contrarian_strategy import ContrarianStrategy
from funding.execution.directional_executor import DirectionalExecutor

logger = logging.getLogger(__name__)
_ENGINE_BUILD_TS = datetime.now(timezone.utc).isoformat()

# Settlement times (UTC hours)
SETTLEMENT_HOURS = {0, 8, 16}


class FundingEngine:
    """Main engine for the funding rate arbitrage system."""

    # Used by enable_contrarian/enable_regime for lazy imports
    _RegimeAdapter = None
    _RegimeSelector = None

    def __init__(
        self,
        on_opportunity: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        on_funding: Optional[Callable] = None,
    ):
        # Clients
        is_paper = config.FUNDING_MODE == "paper"
        futures_url = config.BINANCE_FUTURES_TESTNET_URL if is_paper else config.BINANCE_FUTURES_PROD_URL
        futures_key = config.BINANCE_FUTURES_TESTNET_API_KEY if is_paper else config.BINANCE_FUTURES_API_KEY
        futures_secret = config.BINANCE_FUTURES_TESTNET_API_SECRET if is_paper else config.BINANCE_FUTURES_API_SECRET
        ws_url = config.BINANCE_FUTURES_WS_TESTNET if is_paper else config.BINANCE_FUTURES_WS_PROD

        self._futures_client = BinanceFuturesClient(
            api_key=futures_key,
            api_secret=futures_secret,
            base_url=futures_url,
        )
        spot_url = config.BINANCE_SPOT_TESTNET_URL if is_paper else config.BINANCE_SPOT_PROD_URL
        spot_key = config.BINANCE_SPOT_TESTNET_API_KEY if is_paper else config.BINANCE_SPOT_API_KEY
        spot_secret = config.BINANCE_SPOT_TESTNET_API_SECRET if is_paper else config.BINANCE_SPOT_API_SECRET
        self._spot_client = BinanceSpotClient(
            api_key=spot_key,
            api_secret=spot_secret,
            base_url=spot_url,
        )

        # Data layer
        self._price_cache = FundingPriceCache(max_age_seconds=10)
        self._stream = MarketDataStream(
            ws_url=ws_url,
            price_cache=self._price_cache,
        )
        self._symbol_selector = SymbolSelector(self._futures_client)
        self._validation_mode = bool(config.FUNDING_VALIDATION_MODE)
        self._validation_scope = str(getattr(config, "FUNDING_VALIDATION_SCOPE", "hedge_only")).lower()
        self._validation_run_id = ""
        self._fresh_book_started_at: Optional[str] = None
        self._archived_state_path: Optional[str] = None
        self._execution_mode = (
            "fail_closed"
            if self._validation_mode or config.FUNDING_PAPER_REQUIRE_TESTNET_FILLS
            else "best_effort"
        )
        self._contrarian_disabled_for_validation = (
            self._validation_mode and self._validation_scope == "hedge_only"
        )
        self._online_learner = FundingOnlineLearner(
            watchlist_fn=lambda: self._symbol_selector.watchlist,
        )

        # Callbacks
        self._on_opportunity = on_opportunity
        self._on_trade = on_trade
        self._on_funding = on_funding

        # --- Optional data collectors ---
        self._liquidation_stream: Optional[LiquidationStream] = None
        if config.COLLECT_LIQUIDATIONS:
            self._liquidation_stream = LiquidationStream()
            logger.info("LiquidationStream enabled")

        self._sentiment_collector: Optional[SentimentCollector] = None
        if config.COLLECT_LONG_SHORT:
            self._sentiment_collector = SentimentCollector(
                futures_client=self._futures_client,
                watchlist_fn=lambda: self._symbol_selector.watchlist,
            )
            logger.info("SentimentCollector enabled")

        self._depth_collector: Optional[DepthCollector] = None
        if config.COLLECT_DEPTH:
            self._depth_collector = DepthCollector(
                futures_client=self._futures_client,
                watchlist_fn=lambda: self._symbol_selector.watchlist,
            )
            logger.info("DepthCollector enabled")

        # --- Optional contrarian strategy ---
        self._contrarian_strategy: Optional[ContrarianStrategy] = None
        self._directional_executor: Optional[DirectionalExecutor] = None
        if config.CONTRARIAN_ENABLED and not self._contrarian_disabled_for_validation:
            # Attempt to load a pre-trained model from disk; fall back to None
            # (ContrarianStrategy degrades gracefully when model=None).
            _contrarian_model = None
            try:
                from funding.ml.model_selector import ModelSelector
                _selector = ModelSelector()
                _selector.load_comparison()
                _contrarian_model = _selector.get_model()
                if _contrarian_model is None:
                    from funding.ml.contrarian_baseline import ContrarianBaselineModel

                    _contrarian_model = ContrarianBaselineModel()
                    _contrarian_model.load()
                logger.info(
                    "Contrarian model loaded: %s",
                    type(_contrarian_model).__name__,
                )
            except Exception as _exc:
                logger.warning("Could not load contrarian model from selector: %s", _exc)
                try:
                    from funding.ml.contrarian_baseline import ContrarianBaselineModel

                    _contrarian_model = ContrarianBaselineModel()
                    _contrarian_model.load()
                    logger.info("Using ContrarianBaselineModel fallback")
                except Exception as _fallback_exc:
                    logger.warning(
                        "Could not load contrarian fallback model (%s) — "
                        "ContrarianStrategy will produce no signals until a model is trained",
                        _fallback_exc,
                    )

            self._contrarian_strategy = ContrarianStrategy(model=_contrarian_model)
            self._directional_executor = DirectionalExecutor(
                futures_client=self._futures_client,
            )
            logger.info("Contrarian strategy enabled")

        # --- Optional contrarian online learner ---
        self._contrarian_learner: Optional[ContrarianOnlineLearner] = None
        if config.CONTRARIAN_ENABLED and not self._contrarian_disabled_for_validation:
            self._contrarian_learner = ContrarianOnlineLearner(
                watchlist_fn=lambda: self._symbol_selector.watchlist,
                model_selector=None,   # lazily created inside the learner
                contrarian_strategy=self._contrarian_strategy,
            )
            logger.info("ContrarianOnlineLearner enabled")

        # --- Optional regime adapter ---
        self._regime_adapter = None
        if config.REGIME_ENABLED:
            try:
                from funding.strategy.regime_adapter import RegimeAdapter
                from funding.ml.regime_selector import RegimeSelector
                _regime_selector = RegimeSelector()
                _regime_selector.load_comparison()
                _regime_model = _regime_selector.get_model()
                self._regime_adapter = RegimeAdapter(regime_model=_regime_model)
                logger.info(
                    "RegimeAdapter enabled with model: %s",
                    type(_regime_model).__name__,
                )
            except Exception as _exc:
                logger.warning(
                    "Could not load regime model (%s) — "
                    "RegimeAdapter will use default medium-regime multipliers",
                    _exc,
                )
                from funding.strategy.regime_adapter import RegimeAdapter
                self._regime_adapter = RegimeAdapter(regime_model=None)

        # --- Optional strategy orchestrator ---
        self._orchestrator = None

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._scan_count = 0
        self._opportunity_count = 0
        self._trade_count = 0
        self._last_settlement_hour = -1

        # Contrarian state counters
        self._contrarian_signal_count = 0
        self._contrarian_trade_count = (
            len(self._directional_executor.position_manager.open_positions())
            if self._directional_executor is not None
            else 0
        )
        self._contrarian_cycle_count = 0
        self._contrarian_last_cycle_at: Optional[str] = None
        self._contrarian_last_cycle_diag: Dict[str, object] = {}
        self._learning_events: List[Dict[str, object]] = []
        self._hedge_capital_context: Dict[str, object] = {
            "configured_capital_usd": float(config.FUNDING_MAX_TOTAL_EXPOSURE_USD),
            "deployable_capital_usd": float(config.FUNDING_MAX_TOTAL_EXPOSURE_USD),
            "remaining_exposure_usd": float(config.FUNDING_MAX_TOTAL_EXPOSURE_USD),
            "suggested_position_size_usd": float(
                min(config.FUNDING_MAX_POSITION_USD, config.FUNDING_MAX_TOTAL_EXPOSURE_USD)
            ),
            "capital_source": "configured_cap",
        }
        logger.info(
            "Funding engine boot: mode=%s gate_mode=%s build_ts=%s",
            config.FUNDING_MODE,
            getattr(config, "FUNDING_GATE_MODE", "observe"),
            _ENGINE_BUILD_TS,
        )

    async def start(self) -> None:
        """Start the funding engine."""
        logger.info("Starting funding engine (mode=%s)", config.FUNDING_MODE)
        self._running = True
        risk_manager.reset_circuit_breaker()

        pm = executor.get_position_manager()
        if self._validation_mode and self._validation_scope == "hedge_only":
            self._validation_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            archived = pm.begin_validation_run(
                self._validation_run_id,
                manifest={
                    "engine_build_ts": _ENGINE_BUILD_TS,
                    "mode": config.FUNDING_MODE,
                    "capital_base": float(config.FUNDING_MAX_TOTAL_EXPOSURE_USD),
                    "scope": self._validation_scope,
                    "execution_mode": self._execution_mode,
                },
            )
            validation_context = pm.get_validation_context()
            self._fresh_book_started_at = validation_context.get("fresh_book_started_at")
            self._archived_state_path = str(archived) if archived else None

        # Initialize watchlist
        watchlist = await self._symbol_selector.refresh()
        logger.info("Watchlist: %d symbols", len(watchlist))

        # Start tasks — stored so stop() can cancel them
        self._tasks = [
            asyncio.create_task(self._stream.start(), name="market_data_stream"),
            asyncio.create_task(self._scan_loop(), name="scan_loop"),
            asyncio.create_task(self._funding_settlement_loop(), name="funding_settlement_loop"),
            asyncio.create_task(self._online_learner.run(), name="online_learner"),
        ]

        # Optional collector tasks
        if self._liquidation_stream is not None:
            self._tasks.append(asyncio.create_task(
                self._liquidation_stream.start(), name="liquidation_stream"
            ))

        if self._sentiment_collector is not None:
            self._tasks.append(asyncio.create_task(
                self._sentiment_collector.start(), name="sentiment_collector"
            ))

        if self._depth_collector is not None:
            self._tasks.append(asyncio.create_task(
                self._depth_collector.start(), name="depth_collector"
            ))

        # Optional strategy loops
        if self._contrarian_strategy is not None:
            self._tasks.append(asyncio.create_task(
                self._contrarian_scan_loop(), name="contrarian_scan_loop"
            ))

        if self._contrarian_learner is not None:
            self._tasks.append(asyncio.create_task(
                self._contrarian_learner.run(), name="contrarian_online_learner"
            ))

        if self._regime_adapter is not None:
            self._tasks.append(asyncio.create_task(
                self._regime_update_loop(), name="regime_update_loop"
            ))

        # Strategy orchestrator (autonomous ML lifecycle)
        if config.ORCHESTRATOR_ENABLED:
            try:
                from funding.agents.strategy_orchestrator import StrategyOrchestrator
                self._orchestrator = StrategyOrchestrator(engine=self)
                self._tasks.append(asyncio.create_task(
                    self._orchestrator.run(is_running=lambda: self._running),
                    name="strategy_orchestrator",
                ))
                logger.info("StrategyOrchestrator enabled")
            except Exception as e:
                logger.warning("Could not start StrategyOrchestrator: %s", e)

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the funding engine."""
        self._running = False

        # Cancel all running tasks and wait for them to finish
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        # Component-level cleanup
        self._online_learner.stop()
        if self._contrarian_learner is not None:
            self._contrarian_learner.stop()
        await self._stream.stop()

        # Stop optional collectors
        if self._liquidation_stream is not None:
            await self._liquidation_stream.stop()

        if self._sentiment_collector is not None:
            await self._sentiment_collector.stop()

        if self._depth_collector is not None:
            await self._depth_collector.stop()

        logger.info(
            "Funding engine stopped. Scans=%d, Opportunities=%d, Trades=%d, "
            "ContrarianSignals=%d, ContrarianTrades=%d",
            self._scan_count, self._opportunity_count, self._trade_count,
            self._contrarian_signal_count, self._contrarian_trade_count,
        )

    async def _compute_hedge_capital_context(
        self,
        current_exposure: Decimal,
        open_hedges: int,
    ) -> Dict[str, object]:
        configured_capital = Decimal(str(config.FUNDING_MAX_TOTAL_EXPOSURE_USD))
        spot_balance = Decimal("0")
        futures_wallet_balance = Decimal("0")
        capital_source = "configured_cap"

        try:
            async with async_timeout(10.0):
                spot_balance = await self._spot_client.get_account_balance("USDT")
        except Exception as e:
            logger.debug("Could not fetch spot balance for hedge sizing: %s", e)

        try:
            async with async_timeout(10.0):
                account = await self._futures_client.get_account()
                futures_wallet_balance = Decimal(str(account.get("totalWalletBalance", "0")))
        except Exception as e:
            logger.debug("Could not fetch futures wallet for hedge sizing: %s", e)

        leverage = Decimal(str(max(1, int(config.FUNDING_LEVERAGE))))
        futures_notional_capacity = futures_wallet_balance * leverage
        deployable_capital = configured_capital
        if spot_balance > Decimal("0") and futures_wallet_balance > Decimal("0"):
            deployable_capital = min(configured_capital, spot_balance, futures_notional_capacity)
            capital_source = "wallet_balances"
        elif spot_balance > Decimal("0"):
            deployable_capital = min(configured_capital, spot_balance)
            capital_source = "spot_wallet"
        elif futures_wallet_balance > Decimal("0"):
            deployable_capital = min(configured_capital, futures_notional_capacity)
            capital_source = "futures_wallet"

        remaining_exposure = max(Decimal("0"), deployable_capital - current_exposure)
        remaining_slots = max(0, int(config.FUNDING_MAX_OPEN_HEDGES) - int(open_hedges))
        suggested_position = opportunity_scanner.proportional_position_size(
            remaining_exposure=remaining_exposure,
            remaining_slots=max(1, remaining_slots),
            max_position=config.FUNDING_MAX_POSITION_USD,
        )
        return {
            "configured_capital_usd": float(configured_capital),
            "deployable_capital_usd": float(deployable_capital),
            "spot_balance_usd": float(spot_balance),
            "futures_wallet_balance_usd": float(futures_wallet_balance),
            "futures_notional_capacity_usd": float(futures_notional_capacity),
            "current_exposure_usd": float(current_exposure),
            "remaining_exposure_usd": float(remaining_exposure),
            "remaining_slots": remaining_slots,
            "suggested_position_size_usd": float(suggested_position),
            "capital_source": capital_source,
        }

    @staticmethod
    def _count_settlement_events_between(start: datetime, end: datetime) -> int:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start:
            return 0
        count = 0
        cursor = start.replace(minute=0, second=0, microsecond=0)
        while cursor <= end:
            if cursor > start and cursor <= end and cursor.hour in SETTLEMENT_HOURS:
                count += 1
            cursor += timedelta(hours=1)
        return count

    async def _build_validation_opportunity(
        self,
        opp: FundingOpportunity,
    ) -> Optional[FundingOpportunity]:
        if not self._validation_mode:
            return opp

        rejection = entry_strategy.validation_rejection_reason(opp)
        if rejection is not None:
            executor.get_position_manager().log_rejection(rejection, opp.symbol, {
                "snapshot_age_seconds": str(opp.snapshot_age_seconds),
                "next_funding_time": opp.next_funding_time.isoformat() if opp.next_funding_time else None,
            })
            return None

        try:
            spot_book = await self._spot_client.get_order_book(opp.symbol, limit=20)
            perp_book = await self._futures_client.get_order_book(opp.symbol, limit=20)
        except Exception as exc:
            executor.get_position_manager().log_rejection(
                "missing_spot_market",
                opp.symbol,
                {"error": str(exc)},
            )
            return None

        spot_metrics = symbol_selector.compute_book_metrics(spot_book)
        perp_metrics = symbol_selector.compute_book_metrics(perp_book)
        ok, reason, details = symbol_selector.qualify_symbol_for_trading(
            opp.symbol,
            opp.position_size,
            opp.expected_funding_payment,
            spot_metrics,
            perp_metrics,
        )
        if not ok:
            executor.get_position_manager().log_rejection(reason, opp.symbol, {
                k: str(v) for k, v in details.items()
            })
            return None

        return FundingOpportunity(
            symbol=opp.symbol,
            current_rate=opp.current_rate,
            predicted_rate=opp.predicted_rate,
            annualized_yield=opp.annualized_yield,
            entry_price_spot=spot_metrics["mid"],
            entry_price_perp=perp_metrics["mid"],
            position_size=opp.position_size,
            expected_funding_payment=opp.expected_funding_payment,
            timestamp=opp.timestamp,
            next_funding_time=opp.next_funding_time,
            snapshot_age_seconds=opp.snapshot_age_seconds,
            spot_bid=details["spot_bid"],
            spot_ask=details["spot_ask"],
            perp_bid=details["perp_bid"],
            perp_ask=details["perp_ask"],
            spot_spread_bps=details["spot_spread_bps"],
            perp_spread_bps=details["perp_spread_bps"],
            basis_bps=details["basis_bps"],
            estimated_round_trip_cost_bps=details["estimated_cost_bps"],
            net_expected_edge_usd=details["net_expected_edge_usd"],
            spot_bid_depth_usd=details["spot_bid_depth_usd"],
            spot_ask_depth_usd=details["spot_ask_depth_usd"],
            perp_bid_depth_usd=details["perp_bid_depth_usd"],
            perp_ask_depth_usd=details["perp_ask_depth_usd"],
        )

    async def _qualify_opportunities_for_validation(
        self,
        opportunities: List[FundingOpportunity],
    ) -> List[FundingOpportunity]:
        if not self._validation_mode:
            return opportunities

        qualified: List[dict] = []
        for opp in opportunities[:15]:
            enriched = await self._build_validation_opportunity(opp)
            if enriched is None:
                continue
            qualified.append({
                "opportunity": enriched,
                "net_expected_edge_usd": enriched.net_expected_edge_usd,
                "liquidity_score": min(
                    enriched.spot_bid_depth_usd,
                    enriched.spot_ask_depth_usd,
                    enriched.perp_bid_depth_usd,
                    enriched.perp_ask_depth_usd,
                ),
                "basis_bps": enriched.basis_bps,
                "combined_spread_bps": enriched.spot_spread_bps + enriched.perp_spread_bps,
            })
        ranked = symbol_selector.rank_qualified_opportunities(qualified)
        return [item["opportunity"] for item in ranked]

    async def _scan_loop(self) -> None:
        """Periodic scan for opportunities and execute entries/exits."""
        # Wait for cache to warm up
        await asyncio.sleep(5)

        while self._running:
            try:
                await self._run_scan_cycle()
            except Exception as e:
                logger.exception("Scan cycle error: %s", e)

            await asyncio.sleep(config.FUNDING_POLL_INTERVAL_SECONDS)

    async def _run_scan_cycle(self) -> None:
        """Single scan cycle: detect opportunities → entries → exits."""
        self._scan_count += 1

        # Refresh watchlist periodically
        watchlist = await self._symbol_selector.refresh()
        volume_data = self._symbol_selector.volume_data

        # Get current snapshots
        snapshots = self._price_cache.get_all_snapshots()
        if not snapshots:
            logger.debug("No fresh snapshots in cache")
            return

        # Scan for opportunities
        opportunities = opportunity_scanner.scan_opportunities(
            snapshots, volume_data, watchlist=watchlist
        )
        opportunities = await self._qualify_opportunities_for_validation(opportunities)
        self._opportunity_count += len(opportunities)

        for opp in opportunities:
            if self._on_opportunity:
                self._on_opportunity(opp)

        # Evaluate entries
        pm = executor.get_position_manager()
        open_positions = pm.open_positions()
        current_exposure = sum(p.notional_value() for p in open_positions)
        self._hedge_capital_context = await self._compute_hedge_capital_context(
            current_exposure=current_exposure,
            open_hedges=len(open_positions),
        )
        deployable_capital = Decimal(str(
            self._hedge_capital_context.get("deployable_capital_usd", config.FUNDING_MAX_TOTAL_EXPOSURE_USD)
        ))

        funding_gate_mult = 1.0
        funding_edge_bump = 0.0
        funding_gate_pass = True
        funding_learner_state = self._online_learner.get_state() if self._online_learner else {}
        if funding_learner_state:
            funding_gate_mult = float(funding_learner_state.get("gate_multiplier", 1.0) or 1.0)
            funding_edge_bump = float(funding_learner_state.get("gate_edge_bump", 0.0) or 0.0)
            funding_gate_pass = bool(funding_learner_state.get("strict_gate_pass", False))
            if bool(funding_learner_state.get("prediction_frozen", False)):
                self._learning_events.append(
                    {"kind": "funding_prediction_frozen", "model_id": "funding_online_learner"}
                )
            sat_rate = float(funding_learner_state.get("prediction_saturation_rate", 0.0) or 0.0)
            if sat_rate >= float(config.FUNDING_SATURATION_RATE_THRESHOLD):
                self._learning_events.append(
                    {
                        "kind": "funding_prediction_saturation",
                        "model_id": "funding_online_learner",
                        "rate": round(sat_rate, 4),
                    }
                )
        funding_mode = str(getattr(config, "FUNDING_GATE_MODE", "observe")).lower()
        use_ml = not (funding_mode == "full" and not funding_gate_pass)
        entries = await entry_strategy.evaluate_entries(
            opportunities, open_positions, self._futures_client
            , use_ml=use_ml
            , ml_min_confidence=float(config.FUNDING_ML_MIN_CONFIDENCE) + funding_edge_bump
            , ml_min_predicted_rate=float(config.FUNDING_ML_MIN_PREDICTED_RATE) + funding_edge_bump * 0.001
        )

        planned_exposure = current_exposure
        planned_open_hedges = len(open_positions)
        for opp, size in entries:
            dynamic_size = opportunity_scanner.proportional_position_size(
                remaining_exposure=max(Decimal("0"), deployable_capital - planned_exposure),
                remaining_slots=max(1, int(config.FUNDING_MAX_OPEN_HEDGES) - planned_open_hedges),
                max_position=config.FUNDING_MAX_POSITION_USD,
            )
            size = min(Decimal(str(size)), dynamic_size)
            if size <= Decimal("0"):
                continue
            if funding_gate_mult <= 0:
                continue
            if funding_gate_mult < 1.0:
                size = (Decimal(str(size)) * Decimal(str(funding_gate_mult))).quantize(Decimal("0.01"))
                if size <= Decimal("0"):
                    continue
                opp = FundingOpportunity(
                    symbol=opp.symbol,
                    current_rate=opp.current_rate,
                    predicted_rate=opp.predicted_rate,
                    annualized_yield=opp.annualized_yield,
                    entry_price_spot=opp.entry_price_spot,
                    entry_price_perp=opp.entry_price_perp,
                    position_size=size,
                    expected_funding_payment=opp.expected_funding_payment * Decimal(str(funding_gate_mult)),
                    timestamp=opp.timestamp,
                )
            else:
                opp = FundingOpportunity(
                    symbol=opp.symbol,
                    current_rate=opp.current_rate,
                    predicted_rate=opp.predicted_rate,
                    annualized_yield=opp.annualized_yield,
                    entry_price_spot=opp.entry_price_spot,
                    entry_price_perp=opp.entry_price_perp,
                    position_size=size,
                    expected_funding_payment=(size * opp.current_rate).quantize(Decimal("0.01")),
                    timestamp=opp.timestamp,
                )
            filters = self._symbol_selector.get_exchange_filters(opp.symbol)
            result = await executor.execute_entry(opp, filters)
            if result:
                self._trade_count += 1
                planned_exposure += size
                planned_open_hedges += 1
                if self._on_trade:
                    self._on_trade("entry", result)

        # Evaluate exits
        open_positions = pm.open_positions()  # Refresh after entries
        exits = exit_strategy.evaluate_exits(open_positions, snapshots)

        for symbol in exits:
            result = await executor.execute_exit(symbol)
            if result:
                if result.entry_time is not None and result.exit_time is not None:
                    expected_events = self._count_settlement_events_between(result.entry_time, result.exit_time)
                    missed_events = max(0, expected_events - int(result.realized_funding_events))
                    pm.update_position(
                        symbol,
                        missed_funding_events=missed_events,
                    )
                self._trade_count += 1
                if self._on_trade:
                    self._on_trade("exit", result)

    async def _funding_settlement_loop(self) -> None:
        """Check for funding settlements and record payments."""
        while self._running:
            now = datetime.now(timezone.utc)
            current_hour = now.hour

            if (
                current_hour in SETTLEMENT_HOURS
                and current_hour != self._last_settlement_hour
                and now.minute < 5  # Within 5 min after settlement
            ):
                self._last_settlement_hour = current_hour
                await self._record_funding_settlements()

            await asyncio.sleep(30)

    async def _record_funding_settlements(self) -> None:
        """Record funding payments for all open positions."""
        pm = executor.get_position_manager()
        open_positions = pm.open_positions()

        if not open_positions:
            return

        logger.info("Recording funding settlements for %d positions", len(open_positions))
        funding_info = {}
        try:
            funding_info = await self._futures_client.get_funding_info()
        except Exception as e:
            logger.warning("Failed to fetch funding caps before settlement recording: %s", e)

        for pos in open_positions:
            snapshot = self._price_cache.get_snapshot(pos.symbol)
            if snapshot is None:
                # Fallback: fetch current rate from API
                try:
                    snapshots = await self._futures_client.get_premium_index(pos.symbol)
                    if snapshots:
                        snapshot = snapshots[0]
                except Exception as e:
                    logger.warning("Failed to fetch rate for %s: %s", pos.symbol, e)
                    continue

            if snapshot:
                effective_rate = snapshot.funding_rate
                info = funding_info.get(pos.symbol) if funding_info else None
                cap_applied = False
                if info:
                    rate_cap = Decimal(str(info.get("adjusted_funding_rate_cap", effective_rate)))
                    rate_floor = Decimal(str(info.get("adjusted_funding_rate_floor", effective_rate)))
                    if effective_rate > rate_cap:
                        logger.warning(
                            "Clamping funding rate for %s from %s to cap %s",
                            pos.symbol, effective_rate, rate_cap,
                        )
                        effective_rate = rate_cap
                        cap_applied = True
                    elif effective_rate < rate_floor:
                        logger.warning(
                            "Clamping funding rate for %s from %s to floor %s",
                            pos.symbol, effective_rate, rate_floor,
                        )
                        effective_rate = rate_floor
                        cap_applied = True
                notional = pos.quantity_perp * snapshot.mark_price
                payment = (notional * effective_rate).quantize(Decimal("0.01"))
                pm.record_funding(
                    pos.symbol,
                    payment,
                    expected_payment=payment,
                    cap_applied=cap_applied,
                )
                pm.log_settlement_audit({
                    "symbol": pos.symbol,
                    "position_id": pos.id,
                    "effective_rate": str(effective_rate),
                    "raw_rate": str(snapshot.funding_rate),
                    "payment": str(payment),
                    "cap_applied": cap_applied,
                    "expected_funding_payment": str(pos.expected_funding_payment),
                    "realized_funding_events": int(pos.realized_funding_events),
                    "missed_funding_events": int(pos.missed_funding_events),
                })

                if self._on_funding:
                    self._on_funding(pos.symbol, payment)

    async def _contrarian_scan_loop(self) -> None:
        """Periodic scan for contrarian directional signals and execution."""
        # Wait for cache and collectors to warm up
        await asyncio.sleep(10)

        while self._running:
            try:
                await self._run_contrarian_cycle()
            except Exception as e:
                logger.exception("Contrarian scan cycle error: %s", e)

            await asyncio.sleep(config.CONTRARIAN_SCAN_INTERVAL_SECONDS)

    async def _run_contrarian_cycle(self) -> None:
        """Single contrarian scan cycle: evaluate signals → risk check → execute → check stops."""
        if self._contrarian_strategy is None or self._directional_executor is None:
            self._contrarian_last_cycle_diag = {"reason": "contrarian_disabled"}
            return

        self._contrarian_cycle_count += 1
        self._contrarian_last_cycle_at = datetime.now(timezone.utc).isoformat()
        diag: Dict[str, object] = {
            "cycle": self._contrarian_cycle_count,
            "ts": self._contrarian_last_cycle_at,
        }

        # Get current price snapshots.
        snapshots = self._price_cache.get_all_snapshots()
        diag["snapshots"] = len(snapshots)
        if not snapshots:
            logger.debug("Contrarian cycle: no fresh snapshots in cache")
            diag["reason"] = "no_snapshots"
            self._contrarian_last_cycle_diag = diag
            return

        # Only fetch histories for high-value candidates, otherwise the cycle
        # can starve when the watchlist is large.
        min_rate = config.CONTRARIAN_MIN_FUNDING_RATE
        if str(getattr(config, "FUNDING_MODE", "paper")).lower() == "paper":
            min_rate = min(min_rate, Decimal("0.0002"))
        candidates = [
            (sym, snap)
            for sym, snap in snapshots.items()
            if abs(snap.funding_rate) >= min_rate
        ]
        candidates.sort(key=lambda x: abs(x[1].funding_rate), reverse=True)
        symbol_limit = max(1, int(getattr(config, "CONTRARIAN_HISTORY_SYMBOL_LIMIT", 25)))
        candidate_symbols = [sym for sym, _ in candidates[:symbol_limit]]
        candidate_snapshots = {sym: snapshots[sym] for sym in candidate_symbols}
        diag["min_rate"] = str(min_rate)
        diag["candidate_symbols"] = len(candidate_symbols)

        if not candidate_symbols:
            diag["reason"] = "no_extreme_candidates"
            self._contrarian_last_cycle_diag = diag
            return

        # Gather rate histories with bounded concurrency.
        rate_histories: Dict[str, list] = {}
        concurrency = max(1, int(getattr(config, "CONTRARIAN_HISTORY_FETCH_CONCURRENCY", 8)))
        sem = asyncio.Semaphore(concurrency)

        async def _fetch_history(sym: str):
            async with sem:
                try:
                    async with async_timeout(10.0):
                        history = await self._futures_client.get_funding_rate_history(sym, limit=50)
                    return sym, history
                except Exception as exc:
                    logger.debug("Could not fetch rate history for %s: %s", sym, exc)
                    return sym, None

        fetched = await asyncio.gather(*[_fetch_history(sym) for sym in candidate_symbols])
        for sym, history in fetched:
            if history:
                rate_histories[sym] = history
        diag["history_loaded"] = len(rate_histories)

        # Gather sentiment data from collector if available
        sentiment: Optional[Dict[str, float]] = None
        fear_greed: Optional[int] = None
        if self._sentiment_collector is not None:
            sentiment_state = self._sentiment_collector.get_state()
            # Build symbol -> long_short_ratio map
            ls_data = sentiment_state.get("long_short", {})
            sentiment = {}
            for sym, row in ls_data.items():
                try:
                    ratio_str = row.get("long_short_ratio", "")
                    if ratio_str:
                        sentiment[sym] = float(ratio_str)
                except (ValueError, TypeError):
                    pass

            fg = sentiment_state.get("fear_greed", {})
            try:
                if fg.get("value"):
                    fear_greed = int(fg["value"])
            except (ValueError, TypeError):
                pass

        # Evaluate contrarian signals
        signals = self._contrarian_strategy.evaluate_signals(
            snapshots=candidate_snapshots,
            rate_histories=rate_histories,
            sentiment=sentiment if sentiment else None,
            fear_greed=fear_greed,
        )
        raw_signal_count = len(signals)
        diag["raw_signals"] = raw_signal_count
        if self._contrarian_learner is not None:
            for sig in signals:
                self._contrarian_learner.record_signal_probability(sig.confidence)

        gate_mult = 1.0
        edge_bump = 0.0
        gate_pass = True
        if self._contrarian_learner is not None:
            st = self._contrarian_learner.get_state()
            gate_mult = float(st.get("gate_multiplier", 1.0) or 1.0)
            edge_bump = float(st.get("gate_edge_bump", 0.0) or 0.0)
            gate_pass = bool(st.get("strict_gate_pass", False))
        mode = str(getattr(config, "FUNDING_GATE_MODE", "observe")).lower()
        diag["gate_mode"] = mode
        diag["gate_pass"] = gate_pass
        diag["gate_multiplier"] = round(gate_mult, 4)
        diag["gate_edge_bump"] = round(edge_bump, 6)
        if mode == "full" and not gate_pass:
            signals = []
            diag["gate_result"] = "blocked_full_mode"
        elif edge_bump > 0:
            min_conf = float(config.CONTRARIAN_MIN_CONFIDENCE) + edge_bump
            before = len(signals)
            signals = [s for s in signals if float(s.confidence) >= min_conf]
            diag["edge_filtered"] = before - len(signals)

        self._contrarian_signal_count += len(signals)
        diag["final_signals"] = len(signals)

        if signals:
            logger.info("Contrarian: %d signal(s) generated this cycle", len(signals))

        # For each signal: risk-check, size, execute
        pm = executor.get_position_manager()
        open_hedges = pm.open_positions()
        open_directional = self._directional_executor.position_manager.open_positions()
        opened_positions = 0
        risk_rejections = 0
        risk_reason_counts: Dict[str, int] = {}

        for signal in signals:
            approved, reason = risk_manager.approve_directional(
                signal=signal,
                open_directional=open_directional,
                open_hedges=open_hedges,
            )
            if not approved:
                logger.info(
                    "Contrarian signal %s %s rejected: %s",
                    signal.symbol, signal.direction.value, reason,
                )
                risk_rejections += 1
                key = str(reason)
                risk_reason_counts[key] = int(risk_reason_counts.get(key, 0)) + 1
                continue

            # Fetch account balance for position sizing (best-effort)
            balance = Decimal("0")
            try:
                if hasattr(self._futures_client, "get_account"):
                    async with async_timeout(10.0):
                        account = await self._futures_client.get_account()
                        balance = Decimal(str(account.get("totalWalletBalance", "0")))
            except Exception as e:
                logger.warning(
                    "Could not fetch account balance for contrarian sizing: %s", e
                )
            if balance <= Decimal("0") and str(getattr(config, "FUNDING_MODE", "paper")).lower() == "paper":
                balance = Decimal(str(config.FUNDING_MAX_TOTAL_EXPOSURE_USD))

            params = self._contrarian_strategy.calculate_position_params(
                signal=signal,
                balance=balance,
            )
            if gate_mult < 1.0:
                try:
                    qty = Decimal(str(params.get("quantity", "0")))
                    qty = (qty * Decimal(str(gate_mult))).quantize(Decimal("0.0001"))
                    if qty <= Decimal("0"):
                        continue
                    params["quantity"] = qty
                except Exception:
                    continue

            position = await self._directional_executor.open_position(
                signal=signal,
                params=params,
            )
            if position is not None:
                self._contrarian_trade_count += 1
                opened_positions += 1
                logger.info(
                    "Contrarian position opened: %s %s qty=%s",
                    signal.symbol, signal.direction.value, params.get("quantity"),
                )
                # Refresh open_directional for the next iteration within this cycle
                open_directional = self._directional_executor.position_manager.open_positions()

        # Check stops / take-profits / max-hold for all open directional positions
        closed = await self._directional_executor.check_stops(snapshots)
        diag["risk_rejections"] = risk_rejections
        if risk_reason_counts:
            diag["risk_rejection_reasons"] = risk_reason_counts
        diag["opened_positions"] = opened_positions
        diag["closed_positions"] = len(closed or [])
        if closed:
            logger.info("Contrarian stops triggered, closed positions: %s", closed)
            if self._contrarian_learner is not None:
                dpm = self._directional_executor.position_manager
                all_pos = dpm.all_positions()
                for p in all_pos:
                    if p.symbol in closed and p.realized_pnl is not None and p.entry_price > 0 and p.exit_price > 0:
                        self._contrarian_learner.log_trade_outcome(
                            symbol=p.symbol,
                            direction=p.side.value,
                            entry_price=float(p.entry_price),
                            exit_price=float(p.exit_price),
                        )
        self._contrarian_last_cycle_diag = diag

    async def _regime_update_loop(self) -> None:
        """Periodic regime model update."""
        # Wait for sufficient data to accumulate before first update
        await asyncio.sleep(60)

        while self._running:
            try:
                await self._run_regime_update()
            except Exception as e:
                logger.exception("Regime update error: %s", e)

            await asyncio.sleep(config.REGIME_UPDATE_INTERVAL_HOURS * 3600)

    async def _run_regime_update(self) -> None:
        """Build regime features from persisted data and update the regime adapter."""
        if self._regime_adapter is None:
            return

        try:
            from funding.ml.regime_features import build_regime_features

            watchlist = list(self._symbol_selector.watchlist)
            if not watchlist:
                logger.debug("Regime update: watchlist empty, skipping")
                return

            # build_regime_features is CPU-bound; run in thread to avoid blocking
            features = await asyncio.to_thread(build_regime_features, watchlist)
            if features is None or features.empty:
                logger.warning("Regime update: build_regime_features returned empty DataFrame")
                return

            features = features.dropna()
            if features.empty:
                logger.warning("Regime update: all rows dropped after dropna, skipping")
                return

            regime = self._regime_adapter.update_regime(features)
            logger.info(
                "Regime updated: %d (%s)",
                regime,
                self._regime_adapter.get_state().get("regime_label", "unknown"),
            )
        except Exception as e:
            logger.exception("Regime update cycle failed: %s", e)

    # ------------------------------------------------------------------
    # Lazy activation (called by StrategyOrchestrator at runtime)
    # ------------------------------------------------------------------

    def enable_contrarian(self) -> None:
        """Enable (or hot-swap) the contrarian strategy at runtime.

        If the strategy already exists, reload the model from disk.
        Otherwise create ContrarianStrategy + DirectionalExecutor + OnlineLearner
        and register their async tasks.
        """
        if self._contrarian_disabled_for_validation:
            logger.info("Skipping contrarian enablement because validation scope is hedge-only")
            return
        from funding.ml.model_selector import ModelSelector

        model = None
        model_name = "None"
        try:
            _selector = ModelSelector()
            _selector.load_comparison()
            model = _selector.get_model()
            if model is None:
                from funding.ml.contrarian_baseline import ContrarianBaselineModel

                model = ContrarianBaselineModel()
                model.load()
            model_name = type(model).__name__
        except Exception:
            try:
                from funding.ml.contrarian_baseline import ContrarianBaselineModel

                model = ContrarianBaselineModel()
                model.load()
                model_name = type(model).__name__
            except Exception:
                model = None
                model_name = "None"

        if self._contrarian_strategy is not None:
            # Hot-swap: replace the model on the existing strategy
            self._contrarian_strategy._model = model
            logger.info("Contrarian model hot-swapped to %s", model_name)
            return

        # First-time creation
        self._contrarian_strategy = ContrarianStrategy(model=model)
        self._directional_executor = DirectionalExecutor(
            futures_client=self._futures_client,
        )
        self._contrarian_learner = ContrarianOnlineLearner(
            watchlist_fn=lambda: self._symbol_selector.watchlist,
            model_selector=None,
            contrarian_strategy=self._contrarian_strategy,
        )

        config.CONTRARIAN_ENABLED = True

        # Register tasks if the engine is already running
        if self._running:
            self._tasks.append(asyncio.create_task(
                self._contrarian_scan_loop(), name="contrarian_scan_loop"
            ))
            self._tasks.append(asyncio.create_task(
                self._contrarian_learner.run(), name="contrarian_online_learner"
            ))

        logger.info(
            "Contrarian strategy enabled at runtime with model %s", model_name
        )

    def enable_regime(self) -> None:
        """Enable (or hot-swap) the regime adapter at runtime."""
        from funding.strategy.regime_adapter import RegimeAdapter
        from funding.ml.regime_selector import RegimeSelector

        regime_model = None
        try:
            _selector = RegimeSelector()
            _selector.load_comparison()
            regime_model = _selector.get_model()
        except Exception as e:
            logger.warning("Could not load regime model: %s — using defaults", e)

        if self._regime_adapter is not None:
            # Hot-swap: replace the underlying model
            if regime_model is not None:
                self._regime_adapter._regime_model = regime_model
                logger.info(
                    "Regime model hot-swapped to %s", type(regime_model).__name__
                )
            return

        # First-time creation
        self._regime_adapter = RegimeAdapter(regime_model=regime_model)
        config.REGIME_ENABLED = True

        # Register task if engine is running
        if self._running:
            self._tasks.append(asyncio.create_task(
                self._regime_update_loop(), name="regime_update_loop"
            ))

        logger.info(
            "Regime adapter enabled at runtime with model %s",
            type(regime_model).__name__ if regime_model else "None (default)",
        )

    def get_state(self) -> dict:
        """Get current engine state for dashboard."""
        pm = executor.get_position_manager()
        open_positions = pm.open_positions()
        all_positions = pm.all_positions()
        snapshots = self._price_cache.get_all_snapshots()
        now_utc = datetime.now(timezone.utc)

        # Upcoming 8h funding settlement timing in UTC.
        settlement_hours = sorted(list(SETTLEMENT_HOURS))
        next_settlement = None
        for h in settlement_hours:
            candidate = now_utc.replace(hour=h, minute=0, second=0, microsecond=0)
            if candidate > now_utc:
                next_settlement = candidate
                break
        if next_settlement is None:
            next_settlement = (now_utc.replace(hour=settlement_hours[0], minute=0, second=0, microsecond=0)
                               .replace(day=now_utc.day) + timedelta(days=1))
        minutes_to_settlement = max(0.0, (next_settlement - now_utc).total_seconds() / 60.0)

        # Enrich hedge positions with live funding-rate context.
        enriched_open_positions: List[Dict[str, object]] = []
        est_next_funding_total = Decimal("0")
        est_next_symbols_covered = 0
        for p in open_positions:
            row = p.to_dict()
            row["current_notional"] = str(p.notional_value())
            row["estimated_next_funding"] = None
            row["current_funding_rate"] = None
            snap = snapshots.get(p.symbol)
            if snap is not None:
                notional = p.quantity_perp * snap.mark_price
                est_next = (notional * snap.funding_rate).quantize(Decimal("0.0001"))
                est_next_funding_total += est_next
                est_next_symbols_covered += 1
                row["current_mark_price"] = str(snap.mark_price)
                row["current_funding_rate"] = str(snap.funding_rate)
                row["annualized_yield"] = str((snap.funding_rate * Decimal("3") * Decimal("365")))
                row["current_notional"] = str(notional)
                row["estimated_next_funding"] = str(est_next)
                row["next_funding_time"] = snap.next_funding_time.isoformat()
                row["funding_rate_source"] = "live_cache"
            else:
                row["funding_rate_source"] = "missing_cache"
            if p.entry_time is not None:
                held_h = max(0.0, (now_utc - p.entry_time).total_seconds() / 3600.0)
                row["hold_hours"] = round(held_h, 3)
            enriched_open_positions.append(row)

        # --- Collector states ---
        liquidation_state = (
            self._liquidation_stream.get_stats()
            if self._liquidation_stream is not None
            else None
        )
        sentiment_state = (
            self._sentiment_collector.get_state()
            if self._sentiment_collector is not None
            else None
        )
        depth_state = (
            self._depth_collector.get_state()
            if self._depth_collector is not None
            else None
        )

        # --- Contrarian state ---
        directional_positions = []
        contrarian_win_rate: Optional[float] = None
        contrarian_model_info: Optional[str] = None
        contrarian_wins = 0
        contrarian_losses = 0
        contrarian_total_pnl = Decimal("0")
        contrarian_avg_hold_hours = 0.0
        if self._directional_executor is not None:
            dpm = self._directional_executor.position_manager
            open_directional = dpm.open_positions()
            current_snaps = snapshots
            for p in open_directional:
                row = p.to_dict()
                snap = current_snaps.get(p.symbol)
                current_price = snap.mark_price if snap is not None else p.entry_price
                try:
                    row["current_price"] = str(current_price)
                    upnl = p.unrealized_pnl(current_price)
                    row["unrealized_pnl"] = str(upnl)
                    notional = p.entry_price * p.quantity if p.entry_price > 0 else Decimal("0")
                    row["notional_usd"] = str(notional)
                    row["unrealized_pnl_pct"] = (
                        str((upnl / notional * Decimal("100")) if notional > 0 else Decimal("0"))
                    )
                except Exception:
                    row["current_price"] = str(p.entry_price)
                    row["unrealized_pnl"] = "0"
                    row["notional_usd"] = "0"
                    row["unrealized_pnl_pct"] = "0"
                if p.entry_time is not None:
                    row["hold_hours"] = round(max(0.0, (now_utc - p.entry_time).total_seconds() / 3600.0), 3)
                else:
                    row["hold_hours"] = 0.0
                try:
                    max_hold_h = float(config.CONTRARIAN_MAX_HOLD_HOURS)
                    row["hours_to_max_hold"] = round(max(0.0, max_hold_h - float(row["hold_hours"])), 3)
                except Exception:
                    row["hours_to_max_hold"] = None
                if p.stop_loss > Decimal("0") and p.take_profit > Decimal("0") and current_price > Decimal("0"):
                    if p.side is DirectionalSide.LONG:
                        stop_buffer = ((current_price - p.stop_loss) / current_price) * Decimal("100")
                        tp_buffer = ((p.take_profit - current_price) / current_price) * Decimal("100")
                    else:
                        stop_buffer = ((p.stop_loss - current_price) / current_price) * Decimal("100")
                        tp_buffer = ((current_price - p.take_profit) / current_price) * Decimal("100")
                    row["stop_buffer_pct"] = str(stop_buffer)
                    row["tp_buffer_pct"] = str(tp_buffer)
                directional_positions.append(row)

            # Win rate: closed positions with positive PnL / total closed
            try:
                all_directional = dpm.all_positions()
                closed = [
                    p for p in all_directional
                    if getattr(getattr(p, "status", None), "value", "") not in {"OPEN", "CLOSING"}
                ]
                if closed:
                    contrarian_wins = sum(1 for p in closed if p.realized_pnl > Decimal("0"))
                    contrarian_losses = max(0, len(closed) - contrarian_wins)
                    contrarian_win_rate = contrarian_wins / len(closed)
                    contrarian_total_pnl = sum(
                        (p.realized_pnl - p.trading_fees_paid) for p in closed
                    )
                    holds = [
                        (p.exit_time - p.entry_time).total_seconds() / 3600.0
                        for p in closed
                        if p.entry_time is not None and p.exit_time is not None
                    ]
                    if holds:
                        contrarian_avg_hold_hours = sum(holds) / len(holds)
            except Exception:
                pass

        if self._contrarian_strategy is not None and self._contrarian_strategy._model is not None:
            contrarian_model_info = type(self._contrarian_strategy._model).__name__

        contrarian_state = {
            "enabled": self._contrarian_strategy is not None,
            "signal_count": self._contrarian_signal_count,
            "trade_count": self._contrarian_trade_count,
            "open_positions": directional_positions,
            "positions": directional_positions,
            "win_rate": contrarian_win_rate,
            "model": contrarian_model_info,
            "model_name": contrarian_model_info,
            "wins": contrarian_wins,
            "losses": contrarian_losses,
            "total_pnl": float(contrarian_total_pnl),
            "total_realized_pnl": float(contrarian_total_pnl),
            "avg_hold_hours": round(float(contrarian_avg_hold_hours), 3),
            "cycle_count": self._contrarian_cycle_count,
            "last_cycle_at": self._contrarian_last_cycle_at,
            "last_cycle_diag": self._contrarian_last_cycle_diag,
        }

        # --- Regime state ---
        regime_state = (
            self._regime_adapter.get_state()
            if self._regime_adapter is not None
            else None
        )

        funding_quality = self._online_learner.get_state() if self._online_learner is not None else {}
        contrarian_quality = (
            self._contrarian_learner.get_state() if self._contrarian_learner is not None else {}
        )
        include_contrarian_in_funding_summary = not self._contrarian_disabled_for_validation
        eligible_count = 0
        strict_total = 0
        weighted_wins = 0.0
        weighted_settled = 0.0
        learner_states = [funding_quality]
        if include_contrarian_in_funding_summary:
            learner_states.append(contrarian_quality)
        for learner_state in learner_states:
            if learner_state:
                strict_total += 1
                if bool(learner_state.get("strict_gate_pass", False)):
                    eligible_count += 1
                settled = float(((learner_state.get("rolling_200") or {}).get("settled", 0) or 0))
                wr = float(learner_state.get("prediction_accuracy", 0.0) or 0.0)
                weighted_wins += wr * settled
                weighted_settled += settled
        strict_gate_pass_rate = (eligible_count / strict_total) if strict_total > 0 else 0.0
        weighted_win_rate_pct = (weighted_wins / weighted_settled * 100.0) if weighted_settled > 0 else 0.0

        self._learning_events.extend(self._online_learner.drain_events())
        if self._contrarian_learner is not None:
            self._learning_events.extend(self._contrarian_learner.drain_events())
        self._learning_events = self._learning_events[-200:]

        total_exposure = pm.total_exposure()
        total_funding_collected = sum(p.funding_collected for p in all_positions)
        total_fees_paid = sum(p.trading_fees_paid for p in all_positions)
        realized_net_pnl = total_funding_collected - total_fees_paid
        closed_positions = [p for p in all_positions if p.status == HedgeStatus.CLOSED]
        validation_context = pm.get_validation_context()
        rejection_entries = pm.get_recent_rejections(limit=500)
        rejection_counts: Dict[str, int] = {}
        for entry in rejection_entries:
            reason = str(entry.get("reason", "unknown"))
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        settlement_entries = pm.get_recent_settlement_audit(limit=500)
        simulated_fill_count = sum(1 for p in all_positions if p.fill_source == "simulated")
        stale_open_positions = [
            p for p in open_positions
            if p.entry_time is not None
            and (now_utc - (p.entry_time if p.entry_time.tzinfo else p.entry_time.replace(tzinfo=timezone.utc))).total_seconds() / 3600.0
            > float(config.FUNDING_MAX_HOLD_HOURS)
        ]
        slippage_totals: List[float] = []
        slippage_cost_proxy = Decimal("0")
        basis_drag_proxy = Decimal("0")
        for p in all_positions:
            total_slippage_bps = sum(
                abs(Decimal(str(v)))
                for v in (
                    p.entry_slippage_bps_spot,
                    p.entry_slippage_bps_perp,
                    p.exit_slippage_bps_spot,
                    p.exit_slippage_bps_perp,
                )
            )
            slippage_totals.append(float(total_slippage_bps))
            notional = p.notional_value()
            slippage_cost_proxy += notional * total_slippage_bps / Decimal("10000")
            basis_drag_proxy += notional * (abs(p.entry_basis_bps) + abs(p.exit_basis_bps)) / Decimal("10000")
        avg_realized_slippage_bps = sum(slippage_totals) / len(slippage_totals) if slippage_totals else 0.0
        successful_entries = len(all_positions)
        total_entry_attempts = successful_entries + len(rejection_entries)
        rejection_rate = (len(rejection_entries) / total_entry_attempts) if total_entry_attempts > 0 else 0.0
        realized_funding_events = sum(int(p.realized_funding_events) for p in all_positions)
        missed_funding_events = sum(int(p.missed_funding_events) for p in all_positions)
        cap_applied_count = sum(1 for p in all_positions if bool(p.funding_cap_applied))
        assumed_capital = Decimal(str(
            self._hedge_capital_context.get("deployable_capital_usd", config.FUNDING_MAX_TOTAL_EXPOSURE_USD)
        ))
        capital_base = assumed_capital if assumed_capital > Decimal("0") else max(total_exposure, Decimal("1"))
        deployed_usage_pct = (total_exposure / capital_base * Decimal("100")) if capital_base > Decimal("0") else Decimal("0")
        realized_roi_pct = (realized_net_pnl / capital_base * Decimal("100")) if capital_base > Decimal("0") else Decimal("0")
        projected_next_settlement_pnl = realized_net_pnl + est_next_funding_total
        projected_next_settlement_roi_pct = (
            projected_next_settlement_pnl / capital_base * Decimal("100")
            if capital_base > Decimal("0")
            else Decimal("0")
        )
        execution_quality = {
            "execution_mode": self._execution_mode,
            "simulated_fill_count": simulated_fill_count,
            "avg_realized_slippage_bps": round(avg_realized_slippage_bps, 4),
            "rejection_rate": round(rejection_rate, 4),
            "rejection_count": len(rejection_entries),
            "orphaned_single_leg_incidents": rejection_counts.get("unwind_failed", 0),
            "stale_open_positions": len(stale_open_positions),
            "zero_simulated_fills": simulated_fill_count == 0,
        }
        settlement_audit = {
            "realized_funding_events": realized_funding_events,
            "missed_funding_events": missed_funding_events,
            "funding_cap_applied_count": cap_applied_count,
            "recent": settlement_entries[-20:],
        }
        cost_breakdown = {
            "total_fees_paid_usd": float(total_fees_paid),
            "estimated_slippage_cost_usd": float(slippage_cost_proxy.quantize(Decimal("0.01"))),
            "estimated_basis_drag_usd": float(basis_drag_proxy.quantize(Decimal("0.01"))),
            "total_estimated_friction_usd": float(
                (total_fees_paid + slippage_cost_proxy + basis_drag_proxy).quantize(Decimal("0.01"))
            ),
        }
        paper_rejections = {
            "count": len(rejection_entries),
            "rate": round(rejection_rate, 4),
            "reasons": rejection_counts,
            "recent": rejection_entries[-20:],
        }
        readiness_v2 = None
        try:
            from monitoring.live_readiness import evaluate_binance_live_readiness

            readiness_v2 = evaluate_binance_live_readiness({
                "running": self._running,
                "mode": config.FUNDING_MODE,
                "ws_connected": self._stream.is_connected,
                "trading_halted": risk_manager.trading_halted,
                "online_learner": self._online_learner.get_state(),
                "contrarian_learner": (
                    self._contrarian_learner.get_state() if self._contrarian_learner is not None else None
                ),
                "realized_roi_pct": float(realized_roi_pct),
                "validation_mode": self._validation_mode,
                "validation_scope": self._validation_scope,
                "execution_mode": self._execution_mode,
                "validation_run_id": self._validation_run_id or validation_context.get("validation_run_id"),
                "fresh_book_started_at": self._fresh_book_started_at or validation_context.get("fresh_book_started_at"),
                "contrarian_trading_disabled_for_validation": self._contrarian_disabled_for_validation,
                "execution_quality": execution_quality,
                "settlement_audit": settlement_audit,
                "cost_breakdown": cost_breakdown,
                "paper_rejections": paper_rejections,
                "positions": [p.to_dict() for p in all_positions],
                "realized_net_pnl_usd": float(realized_net_pnl),
                "closed_hedges": len(closed_positions),
            })
        except Exception:
            readiness_v2 = None

        return {
            "mode": config.FUNDING_MODE,
            "running": self._running,
            "ws_connected": self._stream.is_connected,
            "ws_messages": self._stream.message_count,
            "cache_size": self._price_cache.size,
            "watchlist_size": len(self._symbol_selector.watchlist),
            "scan_count": self._scan_count,
            "opportunity_count": self._opportunity_count,
            "trade_count": self._trade_count,
            "open_hedges": len(open_positions),
            "total_exposure": float(total_exposure),
            "total_funding_collected": float(total_funding_collected),
            "total_fees_paid": float(total_fees_paid),
            "assumed_capital_usd": float(capital_base),
            "deployed_usage_pct": float(deployed_usage_pct),
            "realized_net_pnl_usd": float(realized_net_pnl),
            "realized_roi_pct": float(realized_roi_pct),
            "trading_halted": risk_manager.trading_halted,
            "next_settlement_utc": next_settlement.isoformat(),
            "minutes_to_next_settlement": round(minutes_to_settlement, 2),
            "estimated_next_funding_total": float(est_next_funding_total),
            "estimated_next_funding_symbols_covered": est_next_symbols_covered,
            "estimated_next_funding_symbols_total": len(open_positions),
            "projected_next_settlement_pnl_usd": float(projected_next_settlement_pnl),
            "projected_next_settlement_roi_pct": float(projected_next_settlement_roi_pct),
            "positions": enriched_open_positions,
            "all_positions": [p.to_dict() for p in all_positions],
            "hedge_capital_context": self._hedge_capital_context,
            "validation_run_id": self._validation_run_id or validation_context.get("validation_run_id"),
            "validation_mode": self._validation_mode,
            "validation_scope": self._validation_scope,
            "execution_mode": self._execution_mode,
            "fresh_book_started_at": self._fresh_book_started_at or validation_context.get("fresh_book_started_at"),
            "archived_state_path": self._archived_state_path or validation_context.get("archived_state_path"),
            "execution_quality": execution_quality,
            "settlement_audit": settlement_audit,
            "cost_breakdown": cost_breakdown,
            "paper_rejections": paper_rejections,
            "readiness_v2": readiness_v2,
            "closed_hedges": len(closed_positions),
            "online_learner": self._online_learner.get_state(),
            "contrarian_learner": (
                self._contrarian_learner.get_state()
                if self._contrarian_learner is not None
                else None
            ),
            "funding_summary": {
                "eligible_models_count": eligible_count,
                "strict_gate_pass_rate": round(strict_gate_pass_rate, 4),
                "weighted_win_rate_pct": round(weighted_win_rate_pct, 2),
                "strict_gate_pass": bool(strict_total > 0 and eligible_count == strict_total),
                "portfolio_scoped": True,
                "scope": "hedge_only" if self._contrarian_disabled_for_validation else "multi_strategy",
            },
            "learning_events": self._learning_events[-50:],
            "build_info": {
                "git_sha": _git_sha(),
                "started_at": _ENGINE_BUILD_TS,
            },
            "liquidation_stream": liquidation_state,
            "sentiment_collector": sentiment_state,
            "depth_collector": depth_state,
            "contrarian": contrarian_state,
            "contrarian_trading_disabled_for_validation": self._contrarian_disabled_for_validation,
            "regime": regime_state,
            "orchestrator": (
                self._orchestrator.get_state()
                if self._orchestrator is not None
                else None
            ),
        }

    def get_funding_rates(self) -> List[dict]:
        """Get current funding rates for watchlist symbols."""
        snapshots = self._price_cache.get_all_snapshots()
        rates = []
        for symbol in sorted(self._symbol_selector.watchlist):
            snap = snapshots.get(symbol)
            if snap:
                rates.append({
                    "symbol": symbol,
                    "funding_rate": float(snap.funding_rate),
                    "annualized_yield": float(snap.funding_rate * 3 * 365),
                    "mark_price": float(snap.mark_price),
                    "next_funding_time": snap.next_funding_time.isoformat(),
                })
        return rates


async def run_funding_loop(
    on_opportunity: Optional[Callable] = None,
    on_trade: Optional[Callable] = None,
    on_funding: Optional[Callable] = None,
) -> None:
    """Entry point: start the funding engine."""
    engine = FundingEngine(
        on_opportunity=on_opportunity,
        on_trade=on_trade,
        on_funding=on_funding,
    )
    await engine.start()


def _git_sha() -> str:
    try:
        sha = Path(".git/HEAD")
        if not sha.exists():
            return "unknown"
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"
