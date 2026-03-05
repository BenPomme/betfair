"""
Executor router for funding rate arbitrage.
Routes to paper (testnet) or live executor based on FUNDING_MODE config.
Circuit breaker: 3 consecutive failures → trading halted.

OPUS-reviewed: This controls real money routing — a bug here places real orders
in paper mode or vice versa.
"""
import logging
from typing import Optional

import config
from funding.core import risk_manager
from funding.core.schemas import FundingOpportunity, HedgePosition
from funding.execution.paper_executor import FundingPaperExecutor
from funding.execution.position_manager import PositionManager

logger = logging.getLogger(__name__)

_paper_executor: Optional[FundingPaperExecutor] = None
_position_manager: Optional[PositionManager] = None


def get_position_manager() -> PositionManager:
    """Get or create the shared position manager."""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager


def _get_paper_executor() -> FundingPaperExecutor:
    """Get or create the paper executor."""
    global _paper_executor
    if _paper_executor is None:
        _paper_executor = FundingPaperExecutor(
            position_manager=get_position_manager()
        )
    return _paper_executor


async def execute_entry(
    opportunity: FundingOpportunity,
    exchange_filters: Optional[dict] = None,
) -> Optional[HedgePosition]:
    """Execute a hedge entry via the appropriate executor.

    Args:
        opportunity: The funding opportunity to execute.
        exchange_filters: Exchange filters for lot sizing.

    Returns:
        HedgePosition if successful, None if failed.
    """
    if risk_manager.trading_halted:
        logger.warning("Trading halted — not executing entry for %s", opportunity.symbol)
        return None

    mode = config.FUNDING_MODE

    if mode == "paper":
        executor = _get_paper_executor()
        try:
            result = await executor.open_hedge(opportunity, exchange_filters)
            if result:
                risk_manager.record_success()
            else:
                tripped = risk_manager.record_failure()
                if tripped:
                    logger.error("Circuit breaker tripped after failed entry")
            return result
        except Exception as e:
            logger.exception("Paper execution entry failed: %s", e)
            risk_manager.record_failure()
            return None

    elif mode == "live":
        # Phase 3: live execution
        logger.error("Live funding execution not yet implemented")
        return None

    else:
        logger.error("Unknown FUNDING_MODE: %s", mode)
        return None


async def execute_exit(symbol: str) -> Optional[HedgePosition]:
    """Execute a hedge exit via the appropriate executor.

    Args:
        symbol: Symbol to close.

    Returns:
        Updated HedgePosition if successful, None if failed.
    """
    mode = config.FUNDING_MODE

    if mode == "paper":
        executor = _get_paper_executor()
        try:
            result = await executor.close_hedge(symbol)
            if result:
                risk_manager.record_success()
            else:
                risk_manager.record_failure()
            return result
        except Exception as e:
            logger.exception("Paper execution exit failed: %s", e)
            risk_manager.record_failure()
            return None

    elif mode == "live":
        logger.error("Live funding execution not yet implemented")
        return None

    else:
        logger.error("Unknown FUNDING_MODE: %s", mode)
        return None
