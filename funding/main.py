"""
Main event loop for funding rate arbitrage.
Orchestrates: watchlist → WebSocket → scan → entry/exit → funding tracking.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config
from funding.core import opportunity_scanner, risk_manager
from funding.core.schemas import DirectionalSide, FundingOpportunity, FundingSnapshot, HedgePosition
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.data.binance_spot_client import BinanceSpotClient
from funding.data.market_data_stream import MarketDataStream
from funding.data.price_cache import FundingPriceCache
from funding.execution import executor
from funding.ml.online_learner import FundingOnlineLearner
from funding.ml.contrarian_learner import ContrarianOnlineLearner
from funding.strategy import entry_strategy, exit_strategy
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
        self._contrarian_trade_count = (
            len(self._directional_executor.position_manager.open_positions())
            if self._directional_executor is not None
            else 0
        )
        self._contrarian_cycle_count = 0
        self._contrarian_last_cycle_at: Optional[str] = None
        self._contrarian_last_cycle_diag: Dict[str, object] = {}
        self._learning_events: List[Dict[str, object]] = []
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

        for opp, size in entries:
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
        eligible_count = 0
        strict_total = 0
        weighted_wins = 0.0
        weighted_settled = 0.0
        for learner_state in [funding_quality, contrarian_quality]:
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
        assumed_capital = Decimal(str(config.FUNDING_MAX_TOTAL_EXPOSURE_USD))
        capital_base = assumed_capital if assumed_capital > Decimal("0") else max(total_exposure, Decimal("1"))
        deployed_usage_pct = (total_exposure / capital_base * Decimal("100")) if capital_base > Decimal("0") else Decimal("0")
        realized_roi_pct = (realized_net_pnl / capital_base * Decimal("100")) if capital_base > Decimal("0") else Decimal("0")
        projected_next_settlement_pnl = realized_net_pnl + est_next_funding_total
        projected_next_settlement_roi_pct = (
            projected_next_settlement_pnl / capital_base * Decimal("100")
            if capital_base > Decimal("0")
            else Decimal("0")
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
