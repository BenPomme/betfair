"""
Main entry: login -> get markets -> poll prices -> scan -> execute (paper or live).
Runs a single async loop: price poller task + scan/execute loop. No stream required.
"""
import asyncio
import logging
import os
import signal
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, List, Optional

import config
from data.betfair_client import create_and_login
from data.price_cache import PriceCache
from data.price_poller import run_price_poller
from data.market_catalogue import get_market_catalogue, discover_markets
from strategy.market_selector import get_watchlist_market_ids
from core.scanner import scan_market
from core.cross_market_scanner import scan_cross_market
from core.risk_manager import RiskManager
from data.event_grouper import group_by_event, get_cross_market_pairs
from execution.executor import execute_opportunity, init_live_executor
from execution.paper_executor import PaperExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set to False on shutdown so poller and loop exit
_running = True


def _shutdown() -> None:
    global _running
    _running = False


async def run_loop(
    market_ids: List[str],
    price_cache: PriceCache,
    risk_manager: RiskManager,
    paper_executor: PaperExecutor,
    scan_interval_seconds: float = 5.0,
    on_scan: Optional[Callable] = None,
    on_opportunity: Optional[Callable[..., None]] = None,
    on_trade: Optional[Callable[..., None]] = None,
    market_metadata: Optional[dict] = None,
) -> None:
    """Scan/execute loop: every scan_interval_seconds check each market for arbs and execute if allowed."""
    meta = market_metadata or {}

    # Track executed cross-market pairs to avoid repeated trades on same arb
    _executed_cross_pairs: set = set()

    def _execute_opp(opp, on_opportunity_cb, on_trade_cb):
        """Common path: check risk, execute, log."""
        if on_opportunity_cb:
            try:
                on_opportunity_cb(opp)
            except Exception:
                pass
        if not risk_manager.can_execute(opp):
            return
        result = execute_opportunity(opp, paper_executor=paper_executor)
        if result:
            if on_trade_cb:
                try:
                    on_trade_cb(opp, result)
                except Exception:
                    pass
            logger.info(
                "Opportunity executed: market_id=%s net_profit_eur=%s arb_type=%s",
                opp.market_id,
                result.get("net_profit_eur"),
                opp.arb_type,
            )
            risk_manager.register_execution(opp, opp.net_profit_eur)

    while _running:
        # Dynamic stake: fraction of current balance
        effective_stake = (paper_executor.balance * config.STAKE_FRACTION).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        if effective_stake < Decimal("2.00"):
            await asyncio.sleep(scan_interval_seconds)
            continue

        # --- Single-market scans (back-back, lay-lay) ---
        for market_id in market_ids:
            if not _running:
                break
            try:
                m_meta = meta.get(market_id, {})
                event_name = m_meta.get("event_name", "")
                market_start = m_meta.get("market_start")

                if on_scan:
                    try:
                        on_scan(market_id)
                    except Exception:
                        pass
                opp = scan_market(
                    price_cache.get_prices,
                    market_id,
                    event_name=event_name,
                    market_start=market_start,
                    max_stake_eur=effective_stake,
                )
                if opp is not None:
                    _execute_opp(opp, on_opportunity, on_trade)
            except Exception as e:
                logger.exception("Scan/execute error for %s: %s", market_id, e)

        # --- Cross-market scans (MATCH_ODDS vs DRAW_NO_BET) ---
        if config.CROSS_MARKET_ENABLED and meta:
            try:
                event_groups = group_by_event(meta)
                for event_id, event_market_ids in event_groups.items():
                    if not _running:
                        break
                    pairs = get_cross_market_pairs(event_market_ids, meta)
                    for mo_id, dnb_id in pairs:
                        mo_snap = price_cache.get_prices(mo_id)
                        dnb_snap = price_cache.get_prices(dnb_id)
                        if mo_snap is None or dnb_snap is None:
                            continue
                        mo_meta = meta.get(mo_id, {})
                        event_name = mo_meta.get("event_name", "")
                        market_start = mo_meta.get("market_start")
                        pair_key = f"{mo_id}+{dnb_id}"
                        if pair_key in _executed_cross_pairs:
                            continue
                        opp = scan_cross_market(
                            mo_snap, dnb_snap,
                            event_name=event_name,
                            market_start=market_start,
                            max_stake_eur=effective_stake,
                        )
                        if opp is not None:
                            _executed_cross_pairs.add(pair_key)
                            _execute_opp(opp, on_opportunity, on_trade)
            except Exception as e:
                logger.exception("Cross-market scan error: %s", e)

        await asyncio.sleep(scan_interval_seconds)


def main() -> None:
    """Login, get markets, run poller + scan loop. Paper mode only until gate passed."""
    if not config.PAPER_TRADING:
        logger.error("PAPER_TRADING must be true. Refusing to run.")
        return

    # Login
    try:
        client = create_and_login()
    except Exception as e:
        logger.exception("Betfair login failed: %s", e)
        return

    # When PAPER_TRADING=false, call init_live_executor(client) here before starting the loop.

    # Market list: env MARKET_IDS or broad multi-sport discovery
    market_ids_str = os.getenv("MARKET_IDS", "")
    market_ids = [m.strip() for m in market_ids_str.split(",") if m.strip()]
    market_metadata = {}
    runner_names = {}

    if not market_ids:
        try:
            market_ids, market_metadata, runner_names = discover_markets(
                client,
                max_total=config.SCAN_MAX_MARKETS,
                include_in_play=config.SCAN_INCLUDE_IN_PLAY,
            )
        except Exception as e:
            logger.exception("Market discovery failed: %s", e)
            client.logout()
            return

    if not market_ids:
        logger.warning("No markets to watch. Set MARKET_IDS in .env or try when markets are open.")
        client.logout()
        return

    logger.info("Watching %d markets: %s", len(market_ids), market_ids[:5])

    price_cache = PriceCache(max_age_seconds=config.STALE_PRICE_SECONDS)
    risk_manager = RiskManager(
        max_stake_eur=config.MAX_STAKE_EUR,
        daily_loss_limit_eur=config.DAILY_LOSS_LIMIT_EUR,
    )
    paper_executor = PaperExecutor(initial_balance_eur=config.INITIAL_BALANCE_EUR)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _main() -> None:
        poller_task = asyncio.create_task(
            run_price_poller(
                client,
                market_ids,
                price_cache,
                interval_seconds=2.0,
                is_running=lambda: _running,
                runner_names=runner_names,
            )
        )
        loop_task = asyncio.create_task(
            run_loop(
                market_ids,
                price_cache,
                risk_manager,
                paper_executor,
                scan_interval_seconds=5.0,
                market_metadata=market_metadata,
            )
        )
        try:
            await asyncio.gather(poller_task, loop_task)
        except asyncio.CancelledError:
            pass
        finally:
            client.logout()
            logger.info("Shutdown complete.")

    def _signal_handler() -> None:
        _shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(_main())
    except KeyboardInterrupt:
        _shutdown()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
