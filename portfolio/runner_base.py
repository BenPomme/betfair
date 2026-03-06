from __future__ import annotations

import logging
import os
import signal
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import config
from portfolio.accounting import utc_now_iso
from portfolio.ledger import PortfolioLedger
from portfolio.state_store import PortfolioStateStore
from portfolio.types import ModelShadowAccount, PortfolioRunnerSpec, StrategyAccount

logger = logging.getLogger(__name__)


class PortfolioRunnerBase(ABC):
    def __init__(self, spec: PortfolioRunnerSpec):
        self.spec = spec
        self.store = PortfolioStateStore(spec.portfolio_id)
        self.ledger = PortfolioLedger(self.store)
        self._stop_event = threading.Event()
        self._heartbeat_seconds = max(2, int(getattr(config, "PORTFOLIO_RUNNER_HEARTBEAT_SECONDS", 5)))

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            logger.info("%s received signal %s", self.spec.portfolio_id, signum)
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception:
                continue

    def request_stop(self) -> None:
        self._stop_event.set()
        self.store.set_stop_requested()

    def should_stop(self) -> bool:
        return self._stop_event.is_set() or self.store.stop_requested()

    def touch_heartbeat(self, status: str = "running", extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"ts": utc_now_iso(), "status": status, "pid": os.getpid()}
        if extra:
            payload.update(extra)
        self.store.write_heartbeat(payload)

    def initialize_runtime(self) -> None:
        self.store.ensure()
        self.store.clear_stop_requested()
        self.store.write_pid(os.getpid())
        self.store.write_config_snapshot(self.build_config_snapshot())
        self.touch_heartbeat(status="starting")

    def finalize_runtime(self) -> None:
        self.touch_heartbeat(status="stopped")
        self.store.clear_pid()
        self.store.clear_stop_requested()

    def publish_snapshot(
        self,
        *,
        account: StrategyAccount,
        raw_state: Dict[str, Any],
        readiness: Optional[Dict[str, Any]] = None,
        models: Optional[Iterable[ModelShadowAccount]] = None,
        trades: Optional[Iterable[Dict[str, Any]]] = None,
        events: Optional[Iterable[Dict[str, Any]]] = None,
        balance_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.ledger.publish(
            account=account,
            raw_state=raw_state,
            readiness=readiness or {},
            models=models or [],
            trades=trades or [],
            events=events or [],
            balance_history=balance_history or [],
        )
        self.touch_heartbeat(status="running")

    def build_config_snapshot(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.spec.portfolio_id,
            "label": self.spec.label,
            "category": self.spec.category,
            "currency": self.spec.currency,
            "initial_balance": self.spec.initial_balance,
        }

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
