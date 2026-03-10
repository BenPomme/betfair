from __future__ import annotations

import json
import logging
import os
import signal
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config
from factory.manifests import candidate_context_refs_for_portfolio, live_manifest_refs_for_portfolio
from factory.runtime_mode import current_agentic_factory_runtime_mode
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
        self._factory_original_config_values: Dict[str, Any] = {}
        self._factory_applied_overrides: Dict[str, Any] = {}
        self._factory_applied_context: Dict[str, Any] | None = None

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

    def _read_factory_package_payload(self, package_path: str | None) -> Dict[str, Any]:
        if not package_path:
            return {}
        path = Path(package_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _factory_context_sort_key(self, item: Dict[str, Any]) -> tuple[object, ...]:
        source = str(item.get("context_source") or "")
        stage = str(item.get("current_stage") or item.get("approved_stage") or "")
        stage_priority = {
            "approved_live": 0,
            "live_ready": 1,
            "canary_ready": 2,
            "paper": 3,
            "shadow": 4,
            "stress": 5,
            "walkforward": 6,
        }
        return (
            0 if source == "live_manifest" else 1,
            stage_priority.get(stage, 999),
            0 if bool(item.get("strict_gate_pass")) else 1,
            -(float(item.get("fitness_score", 0.0) or 0.0)),
            -(float(item.get("monthly_roi_pct", 0.0) or 0.0)),
            str(item.get("lineage_id") or item.get("manifest_id") or ""),
        )

    def preferred_factory_context(
        self,
        *,
        family_ids: Optional[Iterable[str]] = None,
        include_live: bool = True,
        include_candidates: bool = True,
    ) -> Optional[Dict[str, Any]]:
        runtime_context = self.factory_runtime_context()
        contexts: List[Dict[str, Any]] = []
        if include_live:
            contexts.extend(list(runtime_context.get("factory_live_contexts") or []))
        if include_candidates:
            contexts.extend(list(runtime_context.get("factory_candidate_contexts") or []))
        allowed_families = {str(item) for item in family_ids or [] if str(item)}
        if allowed_families:
            contexts = [item for item in contexts if str(item.get("family_id") or "") in allowed_families]
        if not contexts:
            return None
        contexts.sort(key=self._factory_context_sort_key)
        return dict(contexts[0])

    def apply_factory_config_overrides(
        self,
        overrides: Dict[str, Any],
        *,
        source_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        for attr, value in dict(overrides or {}).items():
            if not hasattr(config, attr):
                continue
            if attr not in self._factory_original_config_values:
                self._factory_original_config_values[attr] = getattr(config, attr)
            setattr(config, attr, value)
            applied[attr] = value
        self._factory_applied_overrides = dict(applied)
        if source_context is not None:
            self._factory_applied_context = {
                "context_source": source_context.get("context_source"),
                "family_id": source_context.get("family_id"),
                "lineage_id": source_context.get("lineage_id"),
                "manifest_id": source_context.get("manifest_id"),
                "current_stage": source_context.get("current_stage") or source_context.get("approved_stage"),
                "fitness_score": source_context.get("fitness_score"),
                "monthly_roi_pct": source_context.get("monthly_roi_pct"),
                "strict_gate_pass": bool(source_context.get("strict_gate_pass")),
            }
        elif not applied:
            self._factory_applied_context = None
        return applied

    def restore_factory_config_overrides(self) -> None:
        for attr, value in self._factory_original_config_values.items():
            setattr(config, attr, value)
        self._factory_original_config_values.clear()
        self._factory_applied_overrides = {}
        self._factory_applied_context = None

    def factory_applied_runtime(self) -> Dict[str, Any]:
        return {
            "factory_applied_context": dict(self._factory_applied_context or {}),
            "factory_applied_overrides": dict(self._factory_applied_overrides),
            "factory_applied_override_count": len(self._factory_applied_overrides),
        }

    def factory_runtime_context(self) -> Dict[str, Any]:
        factory_mode = current_agentic_factory_runtime_mode()
        manifest_refs = live_manifest_refs_for_portfolio(self.spec.portfolio_id)
        candidate_refs = candidate_context_refs_for_portfolio(self.spec.portfolio_id)
        contexts: List[Dict[str, Any]] = []
        package_refs: List[Dict[str, Any]] = []
        runtime_overrides: Dict[str, Dict[str, Any]] = {}
        strategy_families: List[str] = []
        package_payloads: List[Dict[str, Any]] = []
        for item in manifest_refs:
            package_summary = dict(item.get("package") or {})
            package_payload = self._read_factory_package_payload(package_summary.get("package_path"))
            context = {
                "manifest_id": item.get("manifest_id"),
                "family_id": item.get("family_id"),
                "lineage_id": item.get("lineage_id"),
                "approved_at": item.get("approved_at"),
                "artifact_refs": dict(item.get("artifact_refs") or {}),
                "runtime_overrides": dict(item.get("runtime_overrides") or {}),
                "package": package_summary,
                "package_payload": package_payload,
            }
            contexts.append(context)
            if package_summary:
                package_refs.append(package_summary)
            if package_payload:
                package_payloads.append(package_payload)
            family_id = str(item.get("family_id") or "")
            if family_id:
                strategy_families.append(family_id)
                merged_overrides = runtime_overrides.setdefault(family_id, {})
                merged_overrides.update(dict(item.get("runtime_overrides") or {}))
        candidate_package_refs: List[Dict[str, Any]] = []
        candidate_strategy_families: List[str] = []
        candidate_package_payloads: List[Dict[str, Any]] = []
        for item in candidate_refs:
            package_summary = dict(item.get("package") or {})
            package_payload = dict(item.get("package_payload") or {})
            if package_summary:
                candidate_package_refs.append(package_summary)
            if package_payload:
                candidate_package_payloads.append(package_payload)
            family_id = str(item.get("family_id") or "")
            if family_id:
                candidate_strategy_families.append(family_id)
        return {
            "agentic_factory_mode": factory_mode.value,
            "agentic_tokens_allowed": factory_mode.agentic_tokens_allowed,
            "factory_influence_allowed": factory_mode.factory_influence_allowed,
            "factory_live_manifest_count": len(manifest_refs),
            "factory_live_manifests": manifest_refs,
            "factory_live_contexts": contexts,
            "factory_live_artifact_packages": package_refs,
            "factory_live_artifact_payloads": package_payloads,
            "factory_live_strategy_families": sorted(set(strategy_families)),
            "factory_live_runtime_overrides": runtime_overrides,
            "factory_candidate_context_count": len(candidate_refs),
            "factory_candidate_contexts": candidate_refs,
            "factory_candidate_artifact_packages": candidate_package_refs,
            "factory_candidate_artifact_payloads": candidate_package_payloads,
            "factory_candidate_strategy_families": sorted(set(candidate_strategy_families)),
        }

    def build_config_snapshot(self) -> Dict[str, Any]:
        snapshot = {
            "portfolio_id": self.spec.portfolio_id,
            "label": self.spec.label,
            "category": self.spec.category,
            "currency": self.spec.currency,
            "initial_balance": self.spec.initial_balance,
        }
        snapshot.update(self.factory_runtime_context())
        return snapshot

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
