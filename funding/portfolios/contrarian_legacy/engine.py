from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import config
from funding.core.schemas import DirectionalPositionStatus
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.data.market_data_stream import MarketDataStream
from funding.data.price_cache import FundingPriceCache
from funding.data.sentiment_collector import SentimentCollector
from funding.execution.directional_executor import DirectionalExecutor
from funding.execution.directional_position_manager import DirectionalPositionManager
from funding.ml.contrarian_learner import ContrarianOnlineLearner
from funding.ml.model_selector import ModelSelector
from funding.strategy.contrarian_strategy import ContrarianStrategy
from funding.strategy.symbol_selector import SymbolSelector
from funding.utils.async_compat import async_timeout

logger = logging.getLogger(__name__)


class ContrarianLegacyEngine:
    def __init__(
        self,
        *,
        state_path: str,
        trade_log_path: str,
        quality_state_path: str,
    ) -> None:
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
        self._price_cache = FundingPriceCache(max_age_seconds=10)
        self._stream = MarketDataStream(ws_url=ws_url, price_cache=self._price_cache)
        self._symbol_selector = SymbolSelector(self._futures_client)
        self._sentiment_collector = SentimentCollector(
            futures_client=self._futures_client,
            watchlist_fn=lambda: self._symbol_selector.watchlist,
        )
        self._position_manager = DirectionalPositionManager(state_path=state_path)
        self._directional_executor = DirectionalExecutor(
            futures_client=self._futures_client,
            position_manager=self._position_manager,
        )
        self._contrarian_strategy = ContrarianStrategy(model=self._load_model())
        self._contrarian_learner = ContrarianOnlineLearner(
            watchlist_fn=lambda: self._symbol_selector.watchlist,
            model_selector=None,
            contrarian_strategy=self._contrarian_strategy,
            trade_log_path=trade_log_path,
            quality_state_path=quality_state_path,
        )
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._cycle_count = 0
        self._signal_count = 0
        self._trade_count = len(self._position_manager.open_positions())
        self._last_cycle_at: Optional[str] = None
        self._last_cycle_diag: Dict[str, object] = {}
        self._events: List[Dict[str, object]] = []
        self._trade_log_path = Path(trade_log_path)

    def _load_model(self):
        model = None
        try:
            selector = ModelSelector()
            selector.load_comparison()
            model = selector.get_model()
        except Exception as exc:
            logger.warning("Could not load contrarian selector model: %s", exc)
        if model is None:
            try:
                from funding.ml.contrarian_baseline import ContrarianBaselineModel

                model = ContrarianBaselineModel()
                model.load()
            except Exception as exc:
                logger.warning("Could not load contrarian baseline model: %s", exc)
                model = None
        return model

    async def start(self) -> None:
        self._running = True
        watchlist = await self._symbol_selector.refresh()
        logger.info("Contrarian watchlist: %d symbols", len(watchlist))
        self._tasks = [
            asyncio.create_task(self._stream.start(), name="contrarian_market_data_stream"),
            asyncio.create_task(self._sentiment_collector.start(), name="contrarian_sentiment_collector"),
            asyncio.create_task(self._contrarian_scan_loop(), name="contrarian_scan_loop"),
            asyncio.create_task(self._contrarian_learner.run(), name="contrarian_online_learner"),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._contrarian_learner.stop()
        await self._stream.stop()
        await self._sentiment_collector.stop()

    async def _contrarian_scan_loop(self) -> None:
        await asyncio.sleep(10)
        while self._running:
            try:
                await self._run_contrarian_cycle()
            except Exception as exc:
                logger.exception("Contrarian scan cycle error: %s", exc)
                self._events.append(
                    {
                        "kind": "contrarian_cycle_error",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                    }
                )
            await asyncio.sleep(config.CONTRARIAN_SCAN_INTERVAL_SECONDS)

    async def _run_contrarian_cycle(self) -> None:
        self._cycle_count += 1
        self._last_cycle_at = datetime.now(timezone.utc).isoformat()
        diag: Dict[str, object] = {"cycle": self._cycle_count, "ts": self._last_cycle_at}

        snapshots = self._price_cache.get_all_snapshots()
        diag["snapshots"] = len(snapshots)
        if not snapshots:
            diag["reason"] = "no_snapshots"
            self._last_cycle_diag = diag
            return

        min_rate = config.CONTRARIAN_MIN_FUNDING_RATE
        if str(config.FUNDING_MODE).lower() == "paper":
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
        diag["candidate_symbols"] = len(candidate_symbols)
        if not candidate_symbols:
            diag["reason"] = "no_extreme_candidates"
            self._last_cycle_diag = diag
            return

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

        for sym, history in await asyncio.gather(*[_fetch_history(sym) for sym in candidate_symbols]):
            if history:
                rate_histories[sym] = history
        diag["history_loaded"] = len(rate_histories)

        sentiment_state = self._sentiment_collector.get_state()
        long_short = {}
        for sym, row in (sentiment_state.get("long_short") or {}).items():
            try:
                ratio_str = row.get("long_short_ratio")
                if ratio_str:
                    long_short[sym] = float(ratio_str)
            except Exception:
                continue
        fear_greed = None
        try:
            fg = sentiment_state.get("fear_greed") or {}
            if fg.get("value"):
                fear_greed = int(fg.get("value"))
        except Exception:
            fear_greed = None

        signals = self._contrarian_strategy.evaluate_signals(
            snapshots=candidate_snapshots,
            rate_histories=rate_histories,
            sentiment=long_short or None,
            fear_greed=fear_greed,
        )
        for sig in signals:
            self._contrarian_learner.record_signal_probability(sig.confidence)

        gate_state = self._contrarian_learner.get_state()
        gate_mult = float(gate_state.get("gate_multiplier", 1.0) or 1.0)
        edge_bump = float(gate_state.get("gate_edge_bump", 0.0) or 0.0)
        gate_pass = bool(gate_state.get("strict_gate_pass", False))
        mode = str(getattr(config, "FUNDING_GATE_MODE", "observe")).lower()
        if mode == "full" and not gate_pass:
            signals = []
            diag["gate_result"] = "blocked_full_mode"
        elif edge_bump > 0:
            min_conf = float(config.CONTRARIAN_MIN_CONFIDENCE) + edge_bump
            signals = [s for s in signals if float(s.confidence) >= min_conf]
        diag["gate_mode"] = mode
        diag["gate_pass"] = gate_pass
        diag["final_signals"] = len(signals)

        open_directional = self._directional_executor.position_manager.open_positions()
        opened_positions = 0
        risk_rejections = 0
        risk_reason_counts: Dict[str, int] = {}

        for signal in signals:
            if len(open_directional) >= int(config.CONTRARIAN_MAX_POSITIONS):
                risk_rejections += 1
                risk_reason_counts["max_directional"] = risk_reason_counts.get("max_directional", 0) + 1
                continue
            balance = Decimal(str(config.CONTRARIAN_PORTFOLIO_INITIAL_BALANCE_USD))
            params = self._contrarian_strategy.calculate_position_params(signal=signal, balance=balance)
            if gate_mult < 1.0:
                try:
                    qty = Decimal(str(params.get("quantity", "0")))
                    qty = (qty * Decimal(str(gate_mult))).quantize(Decimal("0.0001"))
                    if qty <= Decimal("0"):
                        continue
                    params["quantity"] = qty
                except Exception:
                    continue
            position = await self._directional_executor.open_position(signal=signal, params=params)
            if position is not None:
                opened_positions += 1
                self._trade_count += 1
                open_directional = self._directional_executor.position_manager.open_positions()
                self._events.append(
                    {
                        "kind": "contrarian_position_opened",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "confidence": signal.confidence,
                    }
                )
            else:
                risk_rejections += 1
                risk_reason_counts["execution_failed"] = risk_reason_counts.get("execution_failed", 0) + 1

        closed = await self._directional_executor.check_stops(snapshots)
        if closed:
            for symbol in closed:
                for p in self._position_manager.all_positions():
                    if p.symbol == symbol and p.exit_price > 0 and p.entry_price > 0 and p.status != DirectionalPositionStatus.OPEN:
                        self._contrarian_learner.log_trade_outcome(
                            symbol=p.symbol,
                            direction=p.side.value,
                            entry_price=float(p.entry_price),
                            exit_price=float(p.exit_price),
                        )
                self._events.append(
                    {
                        "kind": "contrarian_position_closed",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                    }
                )

        self._signal_count += len(signals)
        diag["opened_positions"] = opened_positions
        diag["closed_positions"] = len(closed or [])
        diag["risk_rejections"] = risk_rejections
        if risk_reason_counts:
            diag["risk_rejection_reasons"] = risk_reason_counts
        self._last_cycle_diag = diag

    def _trade_log_quality(self) -> Dict[str, object]:
        if not self._trade_log_path.exists():
            return {
                "unique_symbol_count": 0,
                "largest_symbol_share": 0.0,
                "duplicate_signature_rate": 0.0,
            }
        rows = []
        for line in self._trade_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if not rows:
            return {
                "unique_symbol_count": 0,
                "largest_symbol_share": 0.0,
                "duplicate_signature_rate": 0.0,
            }
        symbol_counts = Counter(str(r.get("symbol", "")) for r in rows if r.get("symbol"))
        signature_counts = Counter(
            (
                str(r.get("symbol", "")),
                str(r.get("direction", "")),
                str(r.get("entry_price", "")),
                str(r.get("exit_price", "")),
                str(r.get("pnl_pct", "")),
            )
            for r in rows
        )
        duplicate_total = sum(max(0, count - 1) for count in signature_counts.values())
        largest_symbol_share = max(symbol_counts.values()) / len(rows) if symbol_counts else 0.0
        return {
            "unique_symbol_count": len(symbol_counts),
            "largest_symbol_share": round(largest_symbol_share, 4),
            "duplicate_signature_rate": round(duplicate_total / len(rows), 4),
        }

    def get_state(self) -> Dict[str, object]:
        snapshots = self._price_cache.get_all_snapshots()
        open_positions = self._position_manager.open_positions()
        all_positions = self._position_manager.all_positions()
        now = datetime.now(timezone.utc)
        quality_state = self._contrarian_learner.get_state()
        trade_log_quality = self._trade_log_quality()

        open_rows = []
        for p in open_positions:
            snap = snapshots.get(p.symbol)
            current_price = snap.mark_price if snap is not None else p.entry_price
            unrealized = p.unrealized_pnl(current_price)
            hold_hours = 0.0
            if p.entry_time is not None:
                hold_hours = max(0.0, (now - p.entry_time).total_seconds() / 3600.0)
            open_rows.append(
                {
                    **p.to_dict(),
                    "current_price": str(current_price),
                    "unrealized_pnl": str(unrealized),
                    "notional_usd": str(p.notional_value()),
                    "hold_hours": round(hold_hours, 3),
                }
            )

        closed_positions = [
            p for p in all_positions if p.status not in {DirectionalPositionStatus.OPEN, DirectionalPositionStatus.CLOSING}
        ]
        wins = sum(1 for p in closed_positions if (p.realized_pnl - p.trading_fees_paid) > Decimal("0"))
        losses = sum(1 for p in closed_positions if (p.realized_pnl - p.trading_fees_paid) < Decimal("0"))
        total_realized_pnl = sum((p.realized_pnl - p.trading_fees_paid) for p in closed_positions)
        unrealized_total = sum((p.unrealized_pnl((snapshots.get(p.symbol).mark_price if snapshots.get(p.symbol) else p.entry_price))) for p in open_positions)
        total_fees = sum(p.trading_fees_paid for p in all_positions)
        strict_gate_pass = bool(quality_state.get("strict_gate_pass", False))

        blockers: List[str] = []
        if trade_log_quality["unique_symbol_count"] < int(config.CONTRARIAN_REQUIRE_MIN_UNIQUE_SYMBOLS):
            blockers.append("unique_symbol_count")
        if trade_log_quality["largest_symbol_share"] > float(config.CONTRARIAN_MAX_SYMBOL_CONCENTRATION):
            blockers.append("symbol_concentration")
        if trade_log_quality["duplicate_signature_rate"] > float(config.CONTRARIAN_MAX_DUPLICATE_SIGNATURE_RATE):
            blockers.append("duplicate_signature_rate")
        if not strict_gate_pass:
            blockers.append("strict_gate_pass")
        readiness_status = "paper_validating" if not blockers else "blocked"

        return {
            "portfolio_id": "contrarian_legacy",
            "running": self._running,
            "status": "running" if self._running else "idle",
            "mode": config.FUNDING_MODE,
            "signal_count": self._signal_count,
            "trade_count": len(closed_positions),
            "open_positions": open_rows,
            "positions": open_rows,
            "all_positions": [p.to_dict() for p in all_positions],
            "total_realized_pnl_usd": float(total_realized_pnl),
            "unrealized_pnl_usd": float(unrealized_total),
            "current_balance_usd": float(Decimal(str(config.CONTRARIAN_PORTFOLIO_INITIAL_BALANCE_USD)) + total_realized_pnl + unrealized_total),
            "total_fees_paid_usd": float(total_fees),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(closed_positions)) if closed_positions else 0.0,
            "watchlist_size": len(self._symbol_selector.watchlist),
            "last_cycle_at": self._last_cycle_at,
            "last_cycle_diag": self._last_cycle_diag,
            "online_learner": quality_state,
            "execution_quality": {
                "fill_source": "testnet_or_simulated",
                "open_position_count": len(open_positions),
                "closed_position_count": len(closed_positions),
            },
            "risk": {
                "max_positions": int(config.CONTRARIAN_MAX_POSITIONS),
                "daily_loss_limit_pct": float(config.CONTRARIAN_DAILY_LOSS_LIMIT_PCT),
                "gross_exposure_usd": float(sum(p.notional_value() for p in open_positions)),
            },
            "training_quality": trade_log_quality,
            "quarantined_symbol_count": len(quality_state.get("events", []) or []),
            "readiness": {
                "status": readiness_status,
                "confidence_level": "medium" if strict_gate_pass else "low",
                "blockers": blockers,
                "warnings": [],
                "next_actions": [],
            },
            "events": self._events[-200:] + self._contrarian_learner.drain_events(),
        }
