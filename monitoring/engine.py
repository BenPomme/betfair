"""
Trading engine runnable from the dashboard: start/stop in a background thread,
expose state (P&L, trades, events, scan stats, market breakdown, risk) for the UI.
"""
import asyncio
import importlib
import logging
import os
import sys
import threading
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# Avoid circular import by lazy-importing main
_main_module = None

def _get_main():
    global _main_module
    if _main_module is None:
        import main as m
        _main_module = m
    return _main_module


# Modules to reload on each Start so code changes take effect without
# restarting the whole process.
_RELOAD_MODULES = [
    "core.commission",
    "core.cross_market_scanner",
    "core.scanner",
    "core.risk_manager",
    "core.stake_calculator",
    "execution.paper_executor",
    "execution.executor",
    "execution.live_executor",
    "data.event_grouper",
    "main",
]


def _reload_trading_modules() -> None:
    """Reload core/execution modules so code changes take effect on Start."""
    global _main_module
    for name in _RELOAD_MODULES:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
            except Exception as e:
                logger.warning("Failed to reload %s: %s", name, e)
    # Re-acquire main after reload
    if "main" in sys.modules:
        _main_module = sys.modules["main"]


class TradingEngine:
    """Runs the trading loop in a background thread; exposes state for the dashboard."""

    def __init__(self, max_events: int = 300):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._events: deque = deque(maxlen=max_events)
        self._market_ids: List[str] = []
        self._market_metadata: Dict[str, Dict[str, str]] = {}
        self._price_cache: Any = None
        self._risk_manager: Any = None
        self._paper_executor: Any = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

        # Scan stats
        self._total_scans: int = 0
        self._opportunities_found: int = 0
        self._back_back_found: int = 0
        self._lay_lay_found: int = 0
        self._cross_market_found: int = 0
        self._best_overround: Optional[float] = None
        self._session_start_time: Optional[float] = None

        # Balance history: (timestamp_iso, balance_float) pairs
        self._balance_history: deque = deque(maxlen=500)

    def record_event(self, kind: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append({"kind": kind, "data": data})

    def on_scan(self, market_id: str, overround: Optional[float] = None) -> None:
        """Called after each market scan."""
        with self._lock:
            self._total_scans += 1
            if overround is not None:
                if self._best_overround is None or overround < self._best_overround:
                    self._best_overround = overround
        self.record_event("scan", {"market_id": market_id})

    def on_opportunity(self, opp: Any) -> None:
        """Called when an opportunity is detected."""
        with self._lock:
            self._opportunities_found += 1
            arb_type = getattr(opp, "arb_type", "back_back")
            if arb_type == "cross_market":
                self._cross_market_found += 1
            elif arb_type == "lay_lay":
                self._lay_lay_found += 1
            else:
                self._back_back_found += 1
        self.record_event("opportunity", {
            "market_id": opp.market_id,
            "event_name": opp.event_name,
            "net_profit_eur": float(opp.net_profit_eur),
            "overround": float(opp.overround_raw),
            "arb_type": getattr(opp, "arb_type", "back_back"),
            "selections": [
                {"name": s["name"], "back_price": s.get("back_price"), "lay_price": s.get("lay_price"), "stake_eur": s["stake_eur"]}
                for s in opp.selections
            ],
        })

    def on_trade(self, opp: Any, result: dict) -> None:
        """Called when a trade is executed."""
        with self._lock:
            if self._paper_executor is not None:
                self._balance_history.append((
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    float(self._paper_executor.balance),
                ))
        self.record_event("trade", {
            "market_id": opp.market_id,
            "event_name": opp.event_name,
            "net_profit_eur": float(opp.net_profit_eur),
            "arb_type": getattr(opp, "arb_type", "back_back"),
            "selections": [
                {"name": s["name"], "back_price": s.get("back_price"), "lay_price": s.get("lay_price"), "stake_eur": s["stake_eur"]}
                for s in opp.selections
            ],
        })

    def start(self) -> Dict[str, Any]:
        """Start the trading session in a background thread. Returns status dict."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "error": "Already running"}
            self._error = None
            self._market_ids = []
            self._market_metadata = {}
            self._paper_executor = None
            self._risk_manager = None
            self._price_cache = None
            self._total_scans = 0
            self._opportunities_found = 0
            self._back_back_found = 0
            self._lay_lay_found = 0
            self._cross_market_found = 0
            self._best_overround = None
            self._session_start_time = time.time()
            self._balance_history.clear()

        self._thread = threading.Thread(target=self._run_session, daemon=True)
        self._thread.start()
        return {"ok": True}

    def stop(self) -> Dict[str, Any]:
        """Signal the session to stop (poller and loop will exit)."""
        main = _get_main()
        main._running = False
        return {"ok": True}

    def get_state(self) -> Dict[str, Any]:
        """Current state for the UI: running, balance, P&L, trades, events, markets, scan stats, risk, etc."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            balance = None
            daily_pnl = None
            log_entries: List[dict] = []
            market_ids = list(self._market_ids)
            market_metadata = dict(self._market_metadata)

            if self._paper_executor is not None:
                balance = float(self._paper_executor.balance)
                log_entries = self._paper_executor.log_entries[-50:]
                effective_stake = float(
                    self._paper_executor.balance * config.STAKE_FRACTION
                )
            else:
                effective_stake = None
            if self._risk_manager is not None:
                daily_pnl = float(self._risk_manager._daily_pnl_eur)

            events = list(self._events)

            # Scan stats
            total_scans = self._total_scans
            opportunities_found = self._opportunities_found
            best_overround = self._best_overround
            session_start = self._session_start_time

            # Compute scan rate
            uptime_seconds = 0.0
            scan_rate_per_min = 0.0
            if session_start is not None:
                uptime_seconds = time.time() - session_start
                if uptime_seconds > 0:
                    scan_rate_per_min = (total_scans / uptime_seconds) * 60.0

            # Trade stats
            trade_count = len(log_entries)
            hit_rate = 0.0
            avg_profit = 0.0
            if total_scans > 0:
                hit_rate = (trade_count / total_scans) * 100.0
            if trade_count > 0:
                avg_profit = sum(t.get("net_profit_eur", 0) for t in log_entries) / trade_count

            # Market breakdown
            by_sport: Dict[str, int] = {}  # actually by market_type now
            by_country: Dict[str, int] = {}
            for mid, meta in market_metadata.items():
                market_type = meta.get("market_type") or meta.get("sport_name", "Unknown")
                country = meta.get("country", "") or "Global"
                by_sport[market_type] = by_sport.get(market_type, 0) + 1
                by_country[country] = by_country.get(country, 0) + 1

            # Risk state
            from execution.executor import trading_halted, _consecutive_failures, CIRCUIT_BREAKER_THRESHOLD
            open_bets = 0
            if self._paper_executor is not None:
                open_bets = getattr(self._paper_executor, "_open_bets", 0)

            # Balance history
            balance_history = list(self._balance_history)

        return {
            "running": running,
            "balance_eur": balance,
            "daily_pnl_eur": daily_pnl,
            "trades": log_entries,
            "events": events,
            "market_ids": market_ids,
            "market_count": len(market_ids),
            "error": self._error,
            "scan_stats": {
                "total_scans": total_scans,
                "opportunities_found": opportunities_found,
                "back_back_found": self._back_back_found,
                "lay_lay_found": self._lay_lay_found,
                "cross_market_found": self._cross_market_found,
                "scan_rate_per_min": round(scan_rate_per_min, 1),
                "best_overround": round(best_overround, 6) if best_overround is not None else None,
            },
            "market_breakdown": {
                "by_sport": by_sport,
                "by_country": by_country,
            },
            "risk": {
                "daily_loss_limit": float(config.DAILY_LOSS_LIMIT_EUR),
                "daily_loss_used": abs(daily_pnl) if daily_pnl is not None and daily_pnl < 0 else 0.0,
                "max_stake": float(config.MAX_STAKE_EUR),
                "open_bets": open_bets,
                "circuit_breaker": trading_halted,
                "consecutive_failures": _consecutive_failures,
                "circuit_breaker_threshold": CIRCUIT_BREAKER_THRESHOLD,
            },
            "session": {
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(session_start)) if session_start else None,
                "uptime_seconds": round(uptime_seconds, 0),
                "hit_rate": round(hit_rate, 4),
                "avg_profit": round(avg_profit, 4),
                "trade_count": trade_count,
            },
            "config": {
                "paper_trading": config.PAPER_TRADING,
                "mbr": float(config.MBR),
                "discount_rate": float(config.DISCOUNT_RATE),
                "min_profit": float(config.MIN_NET_PROFIT_EUR),
                "min_liquidity": float(config.MIN_LIQUIDITY_EUR),
                "max_stake": float(config.MAX_STAKE_EUR),
                "daily_loss_limit": float(config.DAILY_LOSS_LIMIT_EUR),
                "sports": config.SCAN_SPORTS,
                "countries": config.SCAN_COUNTRIES,
                "max_markets": config.SCAN_MAX_MARKETS,
                "include_in_play": config.SCAN_INCLUDE_IN_PLAY,
                "initial_balance": float(config.INITIAL_BALANCE_EUR),
                "stake_fraction": float(config.STAKE_FRACTION),
            },
            "balance_history": balance_history,
            "effective_stake": round(effective_stake, 2) if effective_stake is not None else None,
        }

    def _run_session(self) -> None:
        _reload_trading_modules()
        main = _get_main()
        main._running = True
        self._error = None
        client = None
        try:
            from data.betfair_client import create_and_login
            from data.price_cache import PriceCache
            from data.price_poller import run_price_poller
            from core.risk_manager import RiskManager
            from execution.executor import execute_opportunity
            from execution.paper_executor import PaperExecutor

            client = create_and_login()
            from data.market_catalogue import discover_markets

            market_ids_str = os.getenv("MARKET_IDS", "")
            market_ids = [m.strip() for m in market_ids_str.split(",") if m.strip()]
            market_metadata: Dict[str, Dict[str, str]] = {}
            runner_names: Dict[str, Dict[str, str]] = {}

            if not market_ids:
                try:
                    market_ids, market_metadata, runner_names = discover_markets(
                        client,
                        max_total=config.SCAN_MAX_MARKETS,
                        include_in_play=config.SCAN_INCLUDE_IN_PLAY,
                    )
                except Exception as e:
                    logger.warning("Market discovery failed: %s", e)
                    with self._lock:
                        self._error = "Market discovery failed: " + str(e)
                    return

            if not market_ids:
                with self._lock:
                    self._error = "No markets to watch. Set MARKET_IDS in .env or try again later."
                return

            with self._lock:
                self._market_ids = market_ids
                self._market_metadata = market_metadata
                # Record initial balance
                self._balance_history.append((
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    float(config.INITIAL_BALANCE_EUR),
                ))

            price_cache = PriceCache(max_age_seconds=config.STALE_PRICE_SECONDS)
            risk_manager = RiskManager(
                max_stake_eur=config.MAX_STAKE_EUR,
                daily_loss_limit_eur=config.DAILY_LOSS_LIMIT_EUR,
            )
            paper_executor = PaperExecutor(initial_balance_eur=config.INITIAL_BALANCE_EUR)
            with self._lock:
                self._price_cache = price_cache
                self._risk_manager = risk_manager
                self._paper_executor = paper_executor

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _async() -> None:
                poller_task = loop.create_task(
                    run_price_poller(
                        client,
                        market_ids,
                        price_cache,
                        interval_seconds=2.0,
                        is_running=lambda: main._running,
                        runner_names=runner_names,
                    )
                )
                loop_task = loop.create_task(
                    main.run_loop(
                        market_ids,
                        price_cache,
                        risk_manager,
                        paper_executor,
                        scan_interval_seconds=5.0,
                        on_scan=lambda mid: self.on_scan(mid),
                        on_opportunity=lambda opp: self.on_opportunity(opp),
                        on_trade=lambda opp, result: self.on_trade(opp, result),
                        market_metadata=market_metadata,
                    )
                )
                try:
                    await asyncio.gather(poller_task, loop_task)
                except asyncio.CancelledError:
                    pass
                finally:
                    if client:
                        try:
                            client.logout()
                        except Exception:
                            pass

            loop.run_until_complete(_async())
            loop.close()
        except Exception as e:
            logger.exception("Trading session error: %s", e)
            with self._lock:
                self._error = str(e)
            if client:
                try:
                    client.logout()
                except Exception:
                    pass
        finally:
            main._running = False
