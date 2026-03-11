from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class FactoryRuntimeModeView:
    value: str = "hard_stop"
    agentic_tokens_allowed: bool = False
    factory_influence_allowed: bool = False


def current_runtime_mode() -> FactoryRuntimeModeView:
    try:
        from factory.runtime_mode import current_agentic_factory_runtime_mode

        runtime_mode = current_agentic_factory_runtime_mode()
        return FactoryRuntimeModeView(
            value=str(getattr(runtime_mode, "value", "full")),
            agentic_tokens_allowed=bool(getattr(runtime_mode, "agentic_tokens_allowed", False)),
            factory_influence_allowed=bool(getattr(runtime_mode, "factory_influence_allowed", False)),
        )
    except Exception:
        return FactoryRuntimeModeView()


def research_factory_start_blocker(portfolio_id: str) -> str | None:
    try:
        from factory.runtime_mode import research_factory_start_blocker as _blocker

        return _blocker(portfolio_id)
    except Exception:
        return "research_factory_unavailable" if portfolio_id == "research_factory" else None


def live_manifest_refs_for_portfolio(portfolio_id: str) -> List[Dict[str, object]]:
    try:
        from factory.manifests import live_manifest_refs_for_portfolio as _live_refs

        return list(_live_refs(portfolio_id))
    except Exception:
        return []


def candidate_context_refs_for_portfolio(portfolio_id: str) -> List[Dict[str, object]]:
    try:
        from factory.manifests import candidate_context_refs_for_portfolio as _candidate_refs

        return list(_candidate_refs(portfolio_id))
    except Exception:
        return []


def manifest_integration_snapshot(portfolio_id: str) -> Dict[str, Any]:
    runtime_mode = current_runtime_mode()
    live_refs = live_manifest_refs_for_portfolio(portfolio_id)
    candidate_refs = candidate_context_refs_for_portfolio(portfolio_id)
    return {
        "agentic_factory_mode": runtime_mode.value,
        "agentic_tokens_allowed": runtime_mode.agentic_tokens_allowed,
        "factory_influence_allowed": runtime_mode.factory_influence_allowed,
        "factory_live_manifest_count": len(live_refs),
        "factory_live_manifests": live_refs,
        "factory_candidate_context_count": len(candidate_refs),
        "factory_candidate_contexts": candidate_refs,
    }
