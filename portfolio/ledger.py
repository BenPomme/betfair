from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from portfolio.state_store import PortfolioStateStore
from portfolio.types import ModelShadowAccount, StrategyAccount


class PortfolioLedger:
    def __init__(self, store: PortfolioStateStore):
        self._store = store

    def publish(
        self,
        *,
        account: StrategyAccount,
        raw_state: Dict,
        readiness: Optional[Dict] = None,
        models: Optional[Iterable[ModelShadowAccount]] = None,
        trades: Optional[Iterable[Dict]] = None,
        events: Optional[Iterable[Dict]] = None,
        balance_history: Optional[List[Dict]] = None,
    ) -> None:
        self._store.write_account(account)
        if balance_history is not None:
            self._store.write_balance_history(balance_history)
        if trades is not None:
            self._store.write_trades(trades)
        if events is not None:
            self._store.write_events(events)
        if readiness is not None:
            self._store.write_readiness(readiness)
        if models is not None:
            self._store.write_models(models)
        self._store.write_state(raw_state)
