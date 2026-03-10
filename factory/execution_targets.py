from __future__ import annotations

from typing import Iterable


TARGET_PORTFOLIO_ALIASES = {
    "betfair_execution_book": "betfair_core",
    "betfair_prediction_league": "betfair_core",
    "betfair_suspension_lag": "betfair_core",
    "betfair_crossbook_consensus": "betfair_core",
    "betfair_timezone_decay": "betfair_core",
    "polymarket_binary_research": "polymarket_quantum_fold",
}


def resolve_target_portfolio(portfolio_id: str) -> str:
    value = str(portfolio_id or "")
    return TARGET_PORTFOLIO_ALIASES.get(value, value)


def portfolio_target_matches(targets: Iterable[str], portfolio_id: str) -> bool:
    resolved = str(portfolio_id or "")
    return any(resolve_target_portfolio(target) == resolved for target in targets)
