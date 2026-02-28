"""
Executor router: read PAPER_TRADING once at startup; route to paper or live.
Identical interface for both paths. Never override PAPER_TRADING in code.
Circuit breaker: 3 consecutive live failures -> trading_halted, stop sending orders.
"""
import logging
from typing import Any, Optional

import config
from core.types import Opportunity
from execution.paper_executor import PaperExecutor

logger = logging.getLogger(__name__)

# Lazy init when needed (live mode: main must call init_live_executor(client) before loop)
_live_executor = None
_default_paper_executor: Optional[PaperExecutor] = None
_consecutive_failures = 0
CIRCUIT_BREAKER_THRESHOLD = 3
trading_halted = False  # Set True after 3 consecutive execution failures; clear manually


def init_live_executor(client: Any) -> None:
    """Set the Betfair API client for live execution. Call once at startup when PAPER_TRADING=false."""
    global _live_executor
    from execution.live_executor import LiveExecutor
    _live_executor = LiveExecutor(client=client)


def _get_live_executor():
    global _live_executor
    if _live_executor is None:
        from execution.live_executor import LiveExecutor
        _live_executor = LiveExecutor()
    return _live_executor


def execute_opportunity(
    opportunity: Opportunity,
    paper_executor: Optional[PaperExecutor] = None,
) -> Optional[dict]:
    """
    Execute an opportunity: paper path logs and simulates; live path places orders.
    Returns log entry dict in paper mode, None or result in live mode.
    In paper mode, if paper_executor is not passed, uses a shared default instance.
    Circuit breaker: if live and trading_halted, returns None without placing.
    """
    global _consecutive_failures, trading_halted
    if config.PAPER_TRADING:
        if paper_executor is None:
            global _default_paper_executor
            if _default_paper_executor is None:
                _default_paper_executor = PaperExecutor(initial_balance_eur=config.INITIAL_BALANCE_EUR)
            paper_executor = _default_paper_executor
        return paper_executor.log(opportunity)
    else:
        if trading_halted:
            logger.warning("Trading halted after %s consecutive failures; not placing", CIRCUIT_BREAKER_THRESHOLD)
            return None
        try:
            result = _get_live_executor().place(opportunity)
            _consecutive_failures = 0
            return result
        except Exception as e:
            _consecutive_failures += 1
            logger.exception("Live execution failure %s/%s: %s", _consecutive_failures, CIRCUIT_BREAKER_THRESHOLD, e)
            if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                trading_halted = True
                logger.error("Circuit breaker: trading halted. Clear execution.trading_halted to resume.")
                try:
                    from monitoring.alerting import alert_circuit_breaker
                    alert_circuit_breaker()
                except Exception:
                    pass
                raise
