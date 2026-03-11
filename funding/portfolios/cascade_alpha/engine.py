from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
from funding.data.agg_trade_stream import AggTradeStream
from funding.data.binance_futures_client import BinanceFuturesClient
from funding.data.event_feature_buffer import EventFeatureBuffer
from funding.data.liquidation_stream import LiquidationStream
from funding.data.market_data_stream import MarketDataStream
from funding.data.open_interest_poller import OpenInterestPoller
from funding.data.price_cache import FundingPriceCache
from funding.portfolios.cascade_alpha.paper_executor import CascadePaperExecutor
from funding.portfolios.cascade_alpha.risk import CascadeRiskManager
from funding.portfolios.cascade_alpha.signal_engine import CascadeSignalEngine
from funding.strategy.symbol_selector import SymbolSelector, compute_book_metrics

logger = logging.getLogger(__name__)


class CascadeAlphaEngine:
    def __init__(self) -> None:
        self._futures_client = BinanceFuturesClient(
            api_key=config.BINANCE_FUTURES_API_KEY if config.FUNDING_MODE != "paper" else config.BINANCE_FUTURES_TESTNET_API_KEY,
            api_secret=config.BINANCE_FUTURES_API_SECRET if config.FUNDING_MODE != "paper" else config.BINANCE_FUTURES_TESTNET_API_SECRET,
            base_url=config.BINANCE_FUTURES_PROD_URL if config.FUNDING_MODE != "paper" else config.BINANCE_FUTURES_TESTNET_URL,
        )
        self._price_cache = FundingPriceCache(max_age_seconds=10)
        self._stream = MarketDataStream(
            ws_url=config.BINANCE_FUTURES_WS_PROD if config.FUNDING_MODE != "paper" else config.BINANCE_FUTURES_WS_TESTNET,
            price_cache=self._price_cache,
        )
        self._selector = SymbolSelector(self._futures_client)
        self._liquidation_stream = LiquidationStream()
        self._agg_trades = AggTradeStream(
            ws_url=config.BINANCE_FUTURES_WS_PROD if config.FUNDING_MODE != "paper" else config.BINANCE_FUTURES_WS_TESTNET,
            symbols_fn=lambda: self._selector.watchlist,
        )
        self._open_interest = OpenInterestPoller(self._futures_client, symbols_fn=lambda: self._selector.watchlist, interval_seconds=60)
        self._features = EventFeatureBuffer()
        self._signals = CascadeSignalEngine()
        self._risk = CascadeRiskManager()
        self._executor = CascadePaperExecutor()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._events: List[dict] = []
        self._scan_count = 0
        self._signal_count = 0
        self._rejections: dict[str, int] = {}
        self._initial_balance = float(config.CASCADE_ALPHA_INITIAL_BALANCE_USD)
        self._realized_pnl = 0.0
        self._balance_history: List[dict] = []
        self._last_watchlist_refresh: Optional[str] = None
        self._learner_samples: List[dict] = []
        self._learner: Dict[str, object] = {
            "running": bool(getattr(config, "CASCADE_ALPHA_LEARNER_ENABLED", True)),
            "last_update_time": None,
            "settled_count": 0,
            "win_rate_pct": 0.0,
            "avg_net_pnl_usd": 0.0,
            "avg_slippage_bps": 0.0,
            "strict_gate_pass": False,
            "strict_gate_reason": "insufficient_settled_trades",
            "rolling_20": {"settled": 0, "win_rate_pct": 0.0, "avg_net_pnl_usd": 0.0},
            "setup_stats": {},
        }

    def _adaptive_policy(self) -> Dict[str, object]:
        settled = int(self._learner.get("settled_count", 0) or 0)
        activation = int(getattr(config, "CASCADE_ALPHA_POLICY_ACTIVATION_SETTLED", 5))
        strict_gate_pass = bool(self._learner.get("strict_gate_pass", False))
        policy: Dict[str, object] = {
            "mode": "baseline",
            "min_signal_score": 0.0,
            "notional_multiplier": 1.0,
            "max_open_positions": int(config.CASCADE_ALPHA_MAX_OPEN_POSITIONS),
            "max_gross_exposure_usd": float(config.CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD),
            "disabled_setups": [],
        }
        if settled < activation:
            return policy
        if strict_gate_pass:
            policy["mode"] = "validated"
            return policy
        disable_min_settled = max(
            int(getattr(config, "CASCADE_ALPHA_POLICY_ACTIVATION_SETTLED", 5)),
            int(getattr(config, "CASCADE_ALPHA_POLICY_DISABLE_SETUP_MIN_SETTLED", 10)),
        )
        disable_max_avg_net = float(
            getattr(config, "CASCADE_ALPHA_POLICY_DISABLE_SETUP_MAX_AVG_NET_PNL_USD", -0.25)
        )
        disabled_candidates = []
        for setup_name, stats in (self._learner.get("setup_stats") or {}).items():
            try:
                if (
                    int(stats.get("settled", 0) or 0) >= disable_min_settled
                    and float(stats.get("avg_net_pnl_usd", 0.0) or 0.0) <= disable_max_avg_net
                ):
                    disabled_candidates.append(str(setup_name).upper())
            except Exception:
                continue
        setup_keys = {
            str(setup_name).upper()
            for setup_name, stats in (self._learner.get("setup_stats") or {}).items()
            if int((stats or {}).get("settled", 0) or 0) > 0
        }
        disabled_setups = (
            disabled_candidates
            if disabled_candidates and len(disabled_candidates) < len(setup_keys)
            else []
        )
        policy.update(
            {
                "mode": "tightened",
                "min_signal_score": float(getattr(config, "CASCADE_ALPHA_POLICY_STRONG_MIN_SIGNAL_SCORE", 8.0)),
                "notional_multiplier": float(getattr(config, "CASCADE_ALPHA_POLICY_REDUCED_NOTIONAL_MULTIPLIER", 0.5)),
                "max_open_positions": int(getattr(config, "CASCADE_ALPHA_POLICY_REDUCED_MAX_OPEN_POSITIONS", 2)),
                "max_gross_exposure_usd": float(config.CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD)
                * float(getattr(config, "CASCADE_ALPHA_POLICY_REDUCED_NOTIONAL_MULTIPLIER", 0.5)),
                "disabled_setups": disabled_setups,
            }
        )
        return policy

    async def start(self) -> None:
        self._running = True
        await self._selector.refresh()
        self._last_watchlist_refresh = datetime.now(timezone.utc).isoformat()
        self._tasks = [
            asyncio.create_task(self._stream.start(), name="cascade_market_data"),
            asyncio.create_task(self._open_interest.start(), name="cascade_open_interest"),
            asyncio.create_task(self._agg_trades.start(), name="cascade_agg_trades"),
            asyncio.create_task(self._scan_loop(), name="cascade_scan"),
        ]
        try:
            if config.FUNDING_MODE != "paper":
                self._tasks.append(asyncio.create_task(self._liquidation_stream.start(), name="cascade_liquidations"))
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
        await self._stream.stop()
        await self._open_interest.stop()
        await self._agg_trades.stop()
        try:
            await self._liquidation_stream.stop()
        except Exception:
            pass

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                await self._run_scan_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Cascade scan error: %s", exc)
            await asyncio.sleep(15.0)

    def _unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for position in self._executor.open_positions:
            price = current_prices.get(position["symbol"])
            if price is None:
                continue
            side_mult = 1.0 if str(position.get("side", "LONG")).upper() == "LONG" else -1.0
            total += (price - float(position["entry_price"])) * float(position["quantity"]) * side_mult
        return total

    async def _run_scan_cycle(self) -> None:
        self._scan_count += 1
        if self._scan_count % 120 == 1:
            await self._selector.refresh()
            self._last_watchlist_refresh = datetime.now(timezone.utc).isoformat()
        symbols = sorted(self._selector.watchlist)[:8]
        snapshots = self._price_cache.get_all_snapshots()
        if not symbols or not snapshots:
            return
        current_prices = {sym: float(snapshots[sym].mark_price) for sym in symbols if sym in snapshots}
        for position in list(self._executor.open_positions):
            current_price = current_prices.get(position["symbol"])
            if current_price is None:
                continue
            should_close, reason = self._risk.should_force_close(position, current_price)
            if should_close:
                trade = self._executor.close_position(position, current_price, reason)
                self._realized_pnl += float(trade.get("net_pnl_usd", 0.0) or 0.0)
                self._events.append({"kind": "trade_closed", "data": trade})
                self._record_learning_sample(trade)
        gross_exposure = sum(float(pos.get("notional_usd", 0.0) or 0.0) for pos in self._executor.open_positions)
        adaptive_policy = self._adaptive_policy()
        for symbol in symbols:
            snap = snapshots.get(symbol)
            if snap is None:
                continue
            order_book = await self._futures_client.get_order_book(symbol, limit=20)
            metrics = compute_book_metrics(order_book)
            if not metrics:
                continue
            liquidation_events = self._liquidation_stream.get_recent(symbol, minutes=5) if config.FUNDING_MODE != "paper" else []
            liquidation_notional = sum(float(item.get("filled_qty", 0.0) or 0.0) * float(item.get("avg_price", 0.0) or 0.0) for item in liquidation_events)
            oi_row = self._open_interest.get_latest(symbol) or {}
            feature_row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mark_price": float(snap.mark_price),
                "funding_rate": float(snap.funding_rate),
                "open_interest": float(oi_row.get("open_interest", 0.0) or 0.0),
                "spread_bps": float(metrics["spread_bps"]),
                "depth_usd": float(min(metrics["bid_depth_usd"], metrics["ask_depth_usd"])),
                "taker_imbalance": self._agg_trades.get_imbalance(symbol),
                "liquidation_notional_usd": liquidation_notional,
            }
            self._features.add(symbol, feature_row)
            features = self._features.compute_metrics(symbol)
            latest = features.get("latest", {})
            signal = self._signals.classify(
                symbol,
                {
                    **features,
                    "spread_bps": float(metrics["spread_bps"]),
                    "depth_usd": float(min(metrics["bid_depth_usd"], metrics["ask_depth_usd"])),
                    "mark_price": float(snap.mark_price),
                    "taker_imbalance": float(latest.get("taker_imbalance", 0.0) or 0.0),
                },
            )
            if not signal:
                continue
            if str(signal.get("setup", "")).upper() in set(adaptive_policy.get("disabled_setups") or []):
                reason = "setup_disabled_by_learner"
                self._rejections[reason] = self._rejections.get(reason, 0) + 1
                self._events.append({"kind": "signal_rejected", "data": {"symbol": symbol, "reason": reason, "signal": signal}})
                continue
            if float(signal.get("signal_score", 0.0) or 0.0) < float(adaptive_policy.get("min_signal_score", 0.0) or 0.0):
                reason = "signal_score_below_policy"
                self._rejections[reason] = self._rejections.get(reason, 0) + 1
                self._events.append({"kind": "signal_rejected", "data": {"symbol": symbol, "reason": reason, "signal": signal}})
                continue
            approved, reason = self._risk.can_open(
                symbol,
                self._executor.open_positions,
                gross_exposure,
                self._realized_pnl,
                max_open_positions=int(adaptive_policy.get("max_open_positions", int(config.CASCADE_ALPHA_MAX_OPEN_POSITIONS))),
                max_gross_exposure=float(adaptive_policy.get("max_gross_exposure_usd", float(config.CASCADE_ALPHA_MAX_GROSS_EXPOSURE_USD))),
            )
            if not approved:
                self._rejections[reason] = self._rejections.get(reason, 0) + 1
                self._events.append({"kind": "signal_rejected", "data": {"symbol": symbol, "reason": reason, "signal": signal}})
                continue
            notional_multiplier = float(adaptive_policy.get("notional_multiplier", 1.0) or 1.0)
            signal["notional_usd"] = min(
                float(config.CASCADE_ALPHA_MAX_NOTIONAL_PER_TRADE_USD) * notional_multiplier,
                float(config.CASCADE_ALPHA_INITIAL_BALANCE_USD) * 0.2 * notional_multiplier,
            )
            position = self._executor.open_trade(signal)
            self._risk.mark_open(symbol)
            gross_exposure += float(position.get("notional_usd", 0.0) or 0.0)
            self._signal_count += 1
            self._events.append({"kind": "trade_opened", "data": position})
        current_balance = self._initial_balance + self._realized_pnl + self._unrealized_pnl(current_prices)
        if not self._balance_history or abs(float(self._balance_history[-1].get("balance", 0.0)) - current_balance) > 1e-9:
            self._balance_history.append({"ts": datetime.now(timezone.utc).isoformat(), "balance": round(current_balance, 6)})
        self._balance_history = self._balance_history[-1000:]
        self._events = self._events[-300:]

    def _record_learning_sample(self, trade: Dict[str, object]) -> None:
        sample = {
            "trade_id": trade.get("trade_id"),
            "symbol": trade.get("symbol"),
            "setup": trade.get("setup"),
            "side": trade.get("side"),
            "signal_score": float(trade.get("signal_score", 0.0) or 0.0),
            "spread_bps": float(trade.get("spread_bps", 0.0) or 0.0),
            "slippage_bps": float(trade.get("slippage_bps", 0.0) or 0.0),
            "taker_imbalance": float(trade.get("taker_imbalance", 0.0) or 0.0),
            "net_pnl_usd": float(trade.get("net_pnl_usd", 0.0) or 0.0),
            "close_reason": trade.get("close_reason"),
            "closed_at": trade.get("closed_at"),
        }
        self._learner_samples.append(sample)
        self._learner_samples = self._learner_samples[-500:]
        self._update_learner_state()

    def _update_learner_state(self) -> None:
        settled = len(self._learner_samples)
        if settled == 0:
            return
        total_net = sum(float(s.get("net_pnl_usd", 0.0) or 0.0) for s in self._learner_samples)
        wins = sum(1 for s in self._learner_samples if float(s.get("net_pnl_usd", 0.0) or 0.0) >= 0.0)
        avg_slippage = sum(float(s.get("slippage_bps", 0.0) or 0.0) for s in self._learner_samples) / settled
        recent = self._learner_samples[-20:]
        recent_settled = len(recent)
        recent_total_net = sum(float(s.get("net_pnl_usd", 0.0) or 0.0) for s in recent)
        recent_wins = sum(1 for s in recent if float(s.get("net_pnl_usd", 0.0) or 0.0) >= 0.0)
        setup_stats: Dict[str, Dict[str, float]] = {}
        for setup in {"CONTINUATION", "SNAPBACK"}:
            rows = [s for s in self._learner_samples if s.get("setup") == setup]
            if not rows:
                continue
            row_count = len(rows)
            setup_stats[setup.lower()] = {
                "settled": row_count,
                "win_rate_pct": round((sum(1 for s in rows if float(s.get("net_pnl_usd", 0.0) or 0.0) >= 0.0) / row_count) * 100.0, 2),
                "avg_net_pnl_usd": round(sum(float(s.get("net_pnl_usd", 0.0) or 0.0) for s in rows) / row_count, 6),
            }
        min_settled = int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20))
        min_win_rate_pct = float(getattr(config, "CASCADE_ALPHA_MIN_WIN_RATE_PCT", 52))
        min_avg_net = float(getattr(config, "CASCADE_ALPHA_MIN_AVG_NET_PNL_USD", 0))
        strict_gate_pass = settled >= min_settled and ((wins / settled) * 100.0) >= min_win_rate_pct and (total_net / settled) >= min_avg_net
        strict_gate_reason = ""
        if not strict_gate_pass:
            if settled < min_settled:
                strict_gate_reason = "insufficient_settled_trades"
            elif ((wins / settled) * 100.0) < min_win_rate_pct:
                strict_gate_reason = "win_rate_below_threshold"
            else:
                strict_gate_reason = "avg_net_pnl_below_threshold"
        self._learner = {
            "running": bool(getattr(config, "CASCADE_ALPHA_LEARNER_ENABLED", True)),
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "settled_count": settled,
            "win_rate_pct": round((wins / settled) * 100.0, 2),
            "avg_net_pnl_usd": round(total_net / settled, 6),
            "avg_slippage_bps": round(avg_slippage, 6),
            "strict_gate_pass": strict_gate_pass,
            "strict_gate_reason": strict_gate_reason,
            "rolling_20": {
                "settled": recent_settled,
                "win_rate_pct": round((recent_wins / recent_settled) * 100.0, 2) if recent_settled else 0.0,
                "avg_net_pnl_usd": round(recent_total_net / recent_settled, 6) if recent_settled else 0.0,
            },
            "setup_stats": setup_stats,
        }

    def get_state(self) -> dict:
        current_prices = {sym: float(s.mark_price) for sym, s in self._price_cache.get_all_snapshots().items()}
        unrealized = self._unrealized_pnl(current_prices)
        current_balance = self._initial_balance + self._realized_pnl + unrealized
        closed_trades = self._executor.closed_trades
        learner_settled_count = int(self._learner.get("settled_count", 0) or 0)
        settled_count = max(len(closed_trades), learner_settled_count)
        win_rate_pct = float(self._learner.get("win_rate_pct", 0.0) or 0.0)
        avg_net_pnl = float(self._learner.get("avg_net_pnl_usd", 0.0) or 0.0)
        readiness_checks = [
            {
                "name": "engine_running",
                "ok": self._running,
                "reason": "Cascade runner must be active",
            },
            {
                "name": "settled_trades_minimum",
                "ok": settled_count >= int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20)),
                "reason": "Need enough closed trades to evaluate edge",
            },
            {
                "name": "win_rate_threshold",
                "ok": settled_count < int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20))
                or win_rate_pct >= float(getattr(config, "CASCADE_ALPHA_MIN_WIN_RATE_PCT", 52)),
                "reason": "Closed-trade win rate must stay above threshold",
            },
            {
                "name": "avg_net_pnl_threshold",
                "ok": settled_count < int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20))
                or avg_net_pnl >= float(getattr(config, "CASCADE_ALPHA_MIN_AVG_NET_PNL_USD", 0)),
                "reason": "Average net P&L must remain non-negative",
            },
        ]
        readiness_blockers = [c["name"] for c in readiness_checks if not c["ok"]]
        readiness_status = "candidate" if settled_count >= int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20)) and not readiness_blockers else ("paper_validating" if self._running else "stopped")
        readiness = {
            "status": readiness_status,
            "research_only": False,
            "blockers": readiness_blockers if self._running else ["engine_not_running"],
            "confidence": "medium" if settled_count >= int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20)) else "low",
            "score_pct": round((sum(1 for c in readiness_checks if c["ok"]) / len(readiness_checks)) * 100.0, 2),
            "checks": readiness_checks,
        }
        return {
            "portfolio_id": "cascade_alpha",
            "running": self._running,
            "mode": "paper",
            "status": "running" if self._running else "idle",
            "watchlist_size": len(self._selector.watchlist),
            "watchlist": sorted(self._selector.watchlist),
            "last_watchlist_refresh": self._last_watchlist_refresh,
            "scan_count": self._scan_count,
            "signal_count": self._signal_count,
            "trade_count": len(closed_trades),
            "open_positions": self._executor.open_positions,
            "closed_trades": closed_trades[-100:],
            "realized_pnl_usd": round(self._realized_pnl, 6),
            "unrealized_pnl_usd": round(unrealized, 6),
            "current_balance_usd": round(current_balance, 6),
            "fees_paid_usd": round(sum(float(t.get("entry_fee_usd", 0.0) or 0.0) + float(t.get("exit_fee_usd", 0.0) or 0.0) for t in closed_trades), 6),
            "gross_exposure_usd": round(sum(float(p.get("notional_usd", 0.0) or 0.0) for p in self._executor.open_positions), 6),
            "rejections": dict(self._rejections),
            "event_classifier": {
                "continuation_trades": sum(1 for t in closed_trades if t.get("setup") == "CONTINUATION"),
                "snapback_trades": sum(1 for t in closed_trades if t.get("setup") == "SNAPBACK"),
            },
            "execution_quality": {
                "avg_modeled_slippage_bps": round(sum(float(t.get("slippage_bps", 0.0) or 0.0) for t in closed_trades) / len(closed_trades), 6) if closed_trades else 0.0,
                "max_spread_bps": float(config.CASCADE_ALPHA_MAX_SPREAD_BPS),
            },
            "feature_buffer": self._features.get_state(),
            "agg_trade_stream": self._agg_trades.get_state(),
            "open_interest": self._open_interest.get_state(),
            "events": self._events[-200:],
            "balance_history": self._balance_history,
            "readiness": readiness,
            "learner": self._learner,
            "training_progress": {
                "settled_trades": settled_count,
                "min_settled_target": int(getattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 20)),
                "win_rate_pct": win_rate_pct,
                "avg_net_pnl_usd": avg_net_pnl,
            },
            "adaptive_policy": self._adaptive_policy(),
        }
