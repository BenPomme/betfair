from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

import config
from factory.runtime_mode import current_agentic_factory_runtime_mode
from monitoring.portfolio_process_manager import PortfolioProcessManager
from monitoring.portfolio_registry import get_portfolio_spec

_RUNNER_TARGET_ALIASES = {
    "betfair_execution_book": "betfair_core",
    "betfair_prediction_league": "betfair_core",
    "betfair_suspension_lag": "betfair_core",
    "betfair_crossbook_consensus": "betfair_core",
    "betfair_timezone_decay": "betfair_core",
    "polymarket_binary_research": "polymarket_quantum_fold",
}


class FactoryExecutionBridge:
    """Keeps the paper execution plane aligned with active factory families."""

    def __init__(self, process_manager: PortfolioProcessManager | None = None) -> None:
        self._process_manager = process_manager or PortfolioProcessManager()
        self._auto_start = bool(getattr(config, "FACTORY_EXECUTION_AUTOSTART_ENABLED", True))

    def _desired_targets(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        desired: Dict[str, Dict[str, Any]] = {}
        for lineage in state.get("lineages") or []:
            if not lineage.get("active", True):
                continue
            current_stage = str(lineage.get("current_stage") or "")
            if current_stage in {"idea", "spec", "data_check", "goldfish_run", "walkforward", "stress"}:
                continue
            for portfolio_id in lineage.get("target_portfolios") or []:
                requested_portfolio_id = str(portfolio_id)
                resolved_portfolio_id = _RUNNER_TARGET_ALIASES.get(requested_portfolio_id, requested_portfolio_id)
                entry = desired.setdefault(
                    resolved_portfolio_id,
                    {
                        "portfolio_id": resolved_portfolio_id,
                        "requested_targets": set(),
                        "families": set(),
                        "lineages": [],
                        "stages": set(),
                    },
                )
                entry["requested_targets"].add(requested_portfolio_id)
                entry["families"].add(str(lineage.get("family_id") or "unknown"))
                entry["stages"].add(current_stage)
                entry["lineages"].append(
                    {
                        "lineage_id": str(lineage.get("lineage_id") or ""),
                        "role": str(lineage.get("role") or ""),
                        "current_stage": current_stage,
                    }
                )
        rows: List[Dict[str, Any]] = []
        for payload in desired.values():
            rows.append(
                {
                    "portfolio_id": payload["portfolio_id"],
                    "requested_targets": sorted(payload["requested_targets"]),
                    "families": sorted(payload["families"]),
                    "stages": sorted(payload["stages"]),
                    "lineages": list(payload["lineages"]),
                }
            )
        return sorted(rows, key=lambda item: item["portfolio_id"])

    def sync(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime_mode = current_agentic_factory_runtime_mode()
        desired = self._desired_targets(state)
        statuses: List[Dict[str, Any]] = []
        family_counts = defaultdict(int)
        for item in desired:
            for family_id in item["families"]:
                family_counts[family_id] += 1
            portfolio_id = str(item["portfolio_id"])
            status: Dict[str, Any] = {
                "portfolio_id": portfolio_id,
                "families": list(item["families"]),
                "requested_targets": list(item["requested_targets"]),
                "stages": list(item["stages"]),
                "active_lineage_count": len(item["lineages"]),
                "lineage_ids": [row["lineage_id"] for row in item["lineages"]],
                "lineage_roles": sorted({row["role"] for row in item["lineages"] if row.get("role")}),
                "runtime_mode": runtime_mode.value,
                "auto_start_enabled": self._auto_start,
                "desired": True,
            }
            try:
                spec = get_portfolio_spec(portfolio_id)
            except KeyError:
                status.update(
                    {
                        "runner_known": False,
                        "runner_enabled": False,
                        "control_mode": "unknown",
                        "running": False,
                        "status": "monitor_only_target",
                        "note": "Target portfolio is a synthetic monitor view or not registered as a managed runner.",
                    }
                )
                statuses.append(status)
                continue

            status.update(
                {
                    "runner_known": True,
                    "runner_enabled": bool(spec.enabled),
                    "control_mode": spec.control_mode,
                    "label": spec.label,
                    "target_aliases_resolved": list(item["requested_targets"]) != [portfolio_id],
                }
            )
            if spec.control_mode == "disabled":
                status.update(
                    {
                        "running": False,
                        "status": "monitor_only_target",
                        "note": "Target portfolio is a disabled synthetic monitor view, not a runner process.",
                    }
                )
                statuses.append(status)
                continue
            if not spec.enabled:
                status.update(
                    {
                        "running": False,
                        "status": "runner_disabled",
                        "note": "Portfolio exists but is disabled in config, so the factory will not auto-start it.",
                    }
                )
                statuses.append(status)
                continue
            current = self._process_manager.status(portfolio_id)
            status.update(
                {
                    "running": bool(current.get("running")),
                    "pid": current.get("pid"),
                    "heartbeat": current.get("heartbeat"),
                }
            )
            if not runtime_mode.factory_influence_allowed:
                status.update(
                    {
                        "status": "factory_influence_paused",
                        "note": "Runtime mode paused factory influence, so execution auto-start is suppressed.",
                    }
                )
                statuses.append(status)
                continue
            if status["running"]:
                status.update(
                    {
                        "status": "running",
                        "note": "Execution runner is already live for this factory target.",
                    }
                )
                statuses.append(status)
                continue
            if not self._auto_start:
                status.update(
                    {
                        "status": "autostart_disabled",
                        "note": "Execution auto-start is disabled, so the runner must be started manually.",
                    }
                )
                statuses.append(status)
                continue
            started = self._process_manager.start(portfolio_id)
            current = self._process_manager.status(portfolio_id)
            status.update(
                {
                    "running": bool(current.get("running")),
                    "pid": current.get("pid"),
                    "heartbeat": current.get("heartbeat"),
                    "start_result": dict(started),
                }
            )
            if started.get("ok") and current.get("running"):
                status.update(
                    {
                        "status": "started",
                        "note": "Factory auto-started the execution runner for paper validation.",
                    }
                )
            elif started.get("error") == "already_running":
                status.update(
                    {
                        "status": "running",
                        "note": "Execution runner was already running.",
                    }
                )
            else:
                status.update(
                    {
                        "status": "start_failed",
                        "note": str(started.get("error") or "Runner did not start successfully."),
                    }
                )
            statuses.append(status)
        return {
            "auto_start_enabled": self._auto_start,
            "runtime_mode": runtime_mode.value,
            "desired_portfolio_count": len(desired),
            "running_portfolio_count": sum(1 for item in statuses if item.get("running")),
            "family_target_counts": dict(sorted(family_counts.items())),
            "targets": statuses,
        }
