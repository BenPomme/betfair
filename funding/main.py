"""
Main event loop for funding rate arbitrage.
Orchestrates: watchlist → WebSocket → scan → entry/exit → funding tracking.
"""
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Dict, List, Optional

import config
from funding.core import opportunity_scanner, risk_manager
from funding.core.schemas import FundingOpportunity, FundingSnapshot, HedgePosition
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.data.binance_spot_client import BinanceSpotClient
from funding.data.market_data_stream import MarketDataStream
from funding.data.price_cache import FundingPriceCache
from funding.execution import executor
from funding.ml.online_learner import FundingOnlineLearner
from funding.ml.contrarian_learner import ContrarianOnlineLearner
from funding.strategy import entry_strategy, exit_strategy
from funding.strategy.symbol_selector import SymbolSelector

# Optional data collectors — imported unconditionally; instantiated conditionally.
from funding.data.liquidation_stream import LiquidationStream
from funding.data.sentiment_collector import SentimentCollector
from funding.data.depth_collector import DepthCollector

# Optional contrarian strategy components — imported unconditionally; instantiated conditionally.
from funding.strategy.contrarian_strategy import ContrarianStrategy
from funding.execution.directional_executor import DirectionalExecutor

logger = logging.getLogger(__name__)

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

        # Data layer
        self._price_cache = FundingPriceCache(max_age_seconds=10)
        self._stream = MarketDataStream(
            ws_url=ws_url,
            price_cache=self._price_cache,
        )
        self._symbol_selector = SymbolSelector(self._futures_client)
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
        if config.CONTRARIAN_ENABLED:
            # Attempt to load a pre-trained model from disk; fall back to None
            # (ContrarianStrategy degrades gracefully when model=None).
            _contrarian_model = None
            try:
                from funding.ml.model_selector import ModelSelector
                _selector = ModelSelector()
                _selector.load_comparison()
                _contrarian_model = _selector.get_model()
                logger.info(
                    "Contrarian model loaded: %s",
                    type(_contrarian_model).__name__,
                )
            except Exception as _exc:
                logger.warning(
                    "Could not load contrarian model (%s) — "
                    "ContrarianStrategy will produce no signals until a model is trained",
                    _exc,
                )

            self._contrarian_strategy = ContrarianStrategy(model=_contrarian_model)
            self._directional_executor = DirectionalExecutor(
                futures_client=self._futures_client,
            )
            logger.info("Contrarian strategy enabled")

        # --- Optional contrarian online learner ---
        self._contrarian_learner: Optional[ContrarianOnlineLearner] = None
        if config.CONTRARIAN_ENABLED:
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
        self._contrarian_trade_count = 0

    async def start(self) -> None:
        """Start the funding engine."""
        logger.info("Starting funding engine (mode=%s)", config.FUNDING_MODE)
        self._running = True

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
        self._opportunity_count += len(opportunities)

        for opp in opportunities:
            if self._on_opportunity:
                self._on_opportunity(opp)

        # Evaluate entries
        pm = executor.get_position_manager()
        open_positions = pm.open_positions()

        entries = await entry_strategy.evaluate_entries(
            opportunities, open_positions, self._futures_client
        )

        for opp, size in entries:
            filters = self._symbol_selector.get_exchange_filters(opp.symbol)
            result = await executor.execute_entry(opp, filters)
            if result:
                self._trade_count += 1
                if self._on_trade:
                    self._on_trade("entry", result)

        # Evaluate exits
        open_positions = pm.open_positions()  # Refresh after entries
        exits = exit_strategy.evaluate_exits(open_positions, snapshots)

        for symbol in exits:
            result = await executor.execute_exit(symbol)
            if result:
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
                notional = pos.quantity_perp * snapshot.mark_price
                payment = (notional * snapshot.funding_rate).quantize(Decimal("0.01"))
                pm.record_funding(pos.symbol, payment)

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
            return

        # Get current price snapshots
        snapshots = self._price_cache.get_all_snapshots()
        if not snapshots:
            logger.debug("Contrarian cycle: no fresh snapshots in cache")
            return

        # Gather rate histories for all symbols with extreme funding rates
        rate_histories: Dict[str, list] = {}
        for symbol in snapshots:
            try:
                async with asyncio.timeout(10.0):
                    history = await self._futures_client.get_funding_rate_history(
                        symbol, limit=50
                    )
                if history:
                    rate_histories[symbol] = history
            except Exception as e:
                logger.debug("Could not fetch rate history for %s: %s", symbol, e)

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
            snapshots=snapshots,
            rate_histories=rate_histories,
            sentiment=sentiment if sentiment else None,
            fear_greed=fear_greed,
        )
        self._contrarian_signal_count += len(signals)

        if signals:
            logger.info("Contrarian: %d signal(s) generated this cycle", len(signals))

        # For each signal: risk-check, size, execute
        pm = executor.get_position_manager()
        open_hedges = pm.open_positions()
        open_directional = self._directional_executor.position_manager.open_positions()

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
                continue

            # Fetch account balance for position sizing (best-effort)
            balance = Decimal("0")
            try:
                async with asyncio.timeout(10.0):
                    account = await self._futures_client.get_account()
                    balance = Decimal(str(account.get("totalWalletBalance", "0")))
            except Exception as e:
                logger.warning(
                    "Could not fetch account balance for contrarian sizing: %s", e
                )

            params = self._contrarian_strategy.calculate_position_params(
                signal=signal,
                balance=balance,
            )

            position = await self._directional_executor.open_position(
                signal=signal,
                params=params,
            )
            if position is not None:
                self._contrarian_trade_count += 1
                logger.info(
                    "Contrarian position opened: %s %s qty=%s",
                    signal.symbol, signal.direction.value, params.get("quantity"),
                )
                # Refresh open_directional for the next iteration within this cycle
                open_directional = self._directional_executor.position_manager.open_positions()

        # Check stops / take-profits / max-hold for all open directional positions
        closed = await self._directional_executor.check_stops(snapshots)
        if closed:
            logger.info("Contrarian stops triggered, closed positions: %s", closed)

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
        from funding.ml.model_selector import ModelSelector

        _selector = ModelSelector()
        _selector.load_comparison()
        model = _selector.get_model()
        model_name = type(model).__name__

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
        if self._directional_executor is not None:
            dpm = self._directional_executor.position_manager
            open_directional = dpm.open_positions()
            directional_positions = [p.to_dict() for p in open_directional]

            # Win rate: closed positions with positive PnL / total closed
            try:
                all_directional = dpm.all_positions()
                closed = [
                    p for p in all_directional
                    if p.realized_pnl is not None
                ]
                if closed:
                    winners = sum(1 for p in closed if p.realized_pnl > Decimal("0"))
                    contrarian_win_rate = winners / len(closed)
            except Exception:
                pass

        if self._contrarian_strategy is not None and self._contrarian_strategy._model is not None:
            contrarian_model_info = type(self._contrarian_strategy._model).__name__

        contrarian_state = {
            "enabled": self._contrarian_strategy is not None,
            "signal_count": self._contrarian_signal_count,
            "trade_count": self._contrarian_trade_count,
            "open_positions": directional_positions,
            "win_rate": contrarian_win_rate,
            "model": contrarian_model_info,
        }

        # --- Regime state ---
        regime_state = (
            self._regime_adapter.get_state()
            if self._regime_adapter is not None
            else None
        )

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
            "total_exposure": float(pm.total_exposure()),
            "total_funding_collected": float(
                sum(p.funding_collected for p in all_positions)
            ),
            "total_fees_paid": float(
                sum(p.trading_fees_paid for p in all_positions)
            ),
            "trading_halted": risk_manager.trading_halted,
            "positions": [p.to_dict() for p in open_positions],
            "online_learner": self._online_learner.get_state(),
            "contrarian_learner": (
                self._contrarian_learner.get_state()
                if self._contrarian_learner is not None
                else None
            ),
            "liquidation_stream": liquidation_state,
            "sentiment_collector": sentiment_state,
            "depth_collector": depth_state,
            "contrarian": contrarian_state,
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
