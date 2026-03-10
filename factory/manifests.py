from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import config
from factory.contracts import AcceptedStrategyManifest
from factory.registry import FactoryRegistry
from factory.runtime_mode import current_agentic_factory_runtime_mode


def _registry() -> FactoryRegistry:
    return FactoryRegistry(Path(getattr(config, "FACTORY_ROOT", "data/factory")))


def _package_summary(package_path: str | None) -> Dict[str, object]:
    if not package_path:
        return {}
    path = Path(package_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {"package_path": str(path), "package_found": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"package_path": str(path), "package_found": False}
    summary = dict(payload.get("artifact_summary") or {})
    summary["package_path"] = str(path)
    summary["package_found"] = True
    return summary


def live_manifests_for_portfolio(portfolio_id: str) -> List[AcceptedStrategyManifest]:
    if not current_agentic_factory_runtime_mode().factory_influence_allowed:
        return []
    return _registry().live_manifests_for_portfolio(portfolio_id)


def live_manifest_refs_for_portfolio(portfolio_id: str) -> List[Dict[str, object]]:
    return [
        {
            "manifest_id": manifest.manifest_id,
            "family_id": manifest.family_id,
            "lineage_id": manifest.lineage_id,
            "approved_at": manifest.approved_at,
            "approved_stage": manifest.approved_stage,
            "status": manifest.status,
            "artifact_refs": dict(manifest.artifact_refs),
            "runtime_overrides": dict(manifest.runtime_overrides),
            "notes": list(manifest.notes),
            "package": _package_summary(manifest.artifact_refs.get("package")),
        }
        for manifest in live_manifests_for_portfolio(portfolio_id)
    ]


def approve_manifest_for_live(manifest_id: str, *, approved_by: str, note: str | None = None) -> AcceptedStrategyManifest | None:
    return _registry().approve_manifest(manifest_id, approved_by=approved_by, note=note)


def reject_manifest(manifest_id: str, *, note: str | None = None) -> AcceptedStrategyManifest | None:
    return _registry().reject_manifest(manifest_id, note=note)
