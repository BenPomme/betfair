from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import config
from factory.contracts import AcceptedStrategyManifest
from factory.execution_targets import portfolio_target_matches, resolve_target_portfolio
from factory.registry import FactoryRegistry
from factory.runtime_mode import current_agentic_factory_runtime_mode

_CANDIDATE_STAGES = {"shadow", "paper", "canary_ready", "live_ready"}
_STAGE_PRIORITY = {
    "approved_live": 0,
    "live_ready": 1,
    "canary_ready": 2,
    "paper": 3,
    "shadow": 4,
    "stress": 5,
    "walkforward": 6,
}


def _registry() -> FactoryRegistry:
    return FactoryRegistry(Path(getattr(config, "FACTORY_ROOT", "data/factory")))


def _json_path(path_like: str | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _read_json(path_like: str | None) -> Dict[str, Any]:
    path = _json_path(path_like)
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _package_payload(package_path: str | None) -> Dict[str, Any]:
    return _read_json(package_path)


def _package_summary(package_path: str | None) -> Dict[str, object]:
    if not package_path:
        return {}
    path = _json_path(package_path)
    if path is None:
        return {}
    if not path.exists():
        return {"package_path": str(path), "package_found": False}
    payload = _package_payload(str(path))
    if not payload:
        return {"package_path": str(path), "package_found": False}
    summary = dict(payload.get("artifact_summary") or {})
    summary["package_path"] = str(path)
    summary["package_found"] = True
    return summary


def _package_artifact_payloads(package_payload: Dict[str, Any]) -> Dict[str, Any]:
    artifact_payloads: Dict[str, Any] = {}
    for artifact_name, artifact_path in dict(package_payload.get("files") or {}).items():
        artifact_payload = _read_json(str(artifact_path))
        if artifact_payload:
            artifact_payloads[str(artifact_name)] = artifact_payload
    return artifact_payloads


def _prediction_model_kind(requested_model_class: str) -> str | None:
    value = str(requested_model_class or "").strip().lower()
    if not value:
        return None
    if value in {"hybrid_logit", "logit", "probit", "transformer", "lstm", "gru", "sequence"}:
        return "hybrid_logit"
    if value in {"market_calibrated", "calibrated", "xgboost", "gbdt", "tree", "forest"}:
        return "market_calibrated"
    return None


def _strategy_profile(
    row: Dict[str, Any],
    *,
    genome_parameters: Dict[str, Any],
    artifact_payloads: Dict[str, Any],
) -> Dict[str, Any]:
    train_payload = dict(artifact_payloads.get("train") or {})
    features_payload = dict(artifact_payloads.get("features") or {})
    return {
        "selected_feature_subset": genome_parameters.get("selected_feature_subset"),
        "selected_model_class": genome_parameters.get("selected_model_class"),
        "selected_horizon_seconds": genome_parameters.get("selected_horizon_seconds"),
        "selected_lookback_hours": genome_parameters.get("selected_lookback_hours"),
        "selected_min_edge": genome_parameters.get("selected_min_edge"),
        "selected_stake_fraction": genome_parameters.get("selected_stake_fraction"),
        "budget_weight_pct": row.get("budget_weight_pct"),
        "requested_model_class": train_payload.get("requested_model_class"),
        "artifact_min_edge": train_payload.get("min_edge"),
        "resolved_model_engine": train_payload.get("resolved_model_engine"),
        "feature_family": features_payload.get("feature_family"),
        "feature_subset": features_payload.get("feature_subset"),
        "latest_artifact_mode": row.get("latest_artifact_mode"),
    }


def _candidate_context_sort_key(item: Dict[str, Any]) -> tuple[Any, ...]:
    stage = str(item.get("current_stage") or item.get("approved_stage") or "")
    return (
        _STAGE_PRIORITY.get(stage, 999),
        0 if bool(item.get("strict_gate_pass")) else 1,
        -(float(item.get("fitness_score", 0.0) or 0.0)),
        -(float(item.get("monthly_roi_pct", 0.0) or 0.0)),
        str(item.get("lineage_id") or ""),
    )


def live_manifests_for_portfolio(portfolio_id: str) -> List[AcceptedStrategyManifest]:
    if not current_agentic_factory_runtime_mode().factory_influence_allowed:
        return []
    return _registry().live_manifests_for_portfolio(portfolio_id)


def live_manifest_refs_for_portfolio(portfolio_id: str) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    for manifest in live_manifests_for_portfolio(portfolio_id):
        package_payload = _package_payload(manifest.artifact_refs.get("package"))
        refs.append(
            {
                "context_source": "live_manifest",
            "manifest_id": manifest.manifest_id,
            "family_id": manifest.family_id,
            "lineage_id": manifest.lineage_id,
            "approved_at": manifest.approved_at,
            "approved_stage": manifest.approved_stage,
                "current_stage": manifest.approved_stage,
            "status": manifest.status,
            "artifact_refs": dict(manifest.artifact_refs),
            "runtime_overrides": dict(manifest.runtime_overrides),
            "notes": list(manifest.notes),
            "package": _package_summary(manifest.artifact_refs.get("package")),
                "package_payload": package_payload,
                "artifact_payloads": _package_artifact_payloads(package_payload),
                "strict_gate_pass": True,
                "fitness_score": None,
                "monthly_roi_pct": None,
            }
        )
    return sorted(refs, key=_candidate_context_sort_key)


def candidate_context_refs_for_portfolio(portfolio_id: str) -> List[Dict[str, object]]:
    if not current_agentic_factory_runtime_mode().factory_influence_allowed:
        return []
    state = _registry().read_state()
    rows = list(state.get("lineages") or [])
    refs: List[Dict[str, object]] = []
    for row in rows:
        if not row.get("active", True):
            continue
        current_stage = str(row.get("current_stage") or "")
        if current_stage not in _CANDIDATE_STAGES:
            continue
        target_portfolios = list(row.get("target_portfolios") or [])
        if not portfolio_target_matches(target_portfolios, portfolio_id):
            continue
        lineage_id = str(row.get("lineage_id") or "")
        genome = _registry().load_genome(lineage_id)
        genome_parameters = dict((genome.parameters if genome else {}) or {})
        package_path = str(row.get("latest_artifact_package") or "")
        package_payload = _package_payload(package_path)
        artifact_payloads = _package_artifact_payloads(package_payload)
        refs.append(
            {
                "context_source": "candidate_lineage",
                "family_id": str(row.get("family_id") or ""),
                "lineage_id": lineage_id,
                "label": row.get("label"),
                "current_stage": current_stage,
                "role": row.get("role"),
                "iteration_status": row.get("iteration_status"),
                "strict_gate_pass": bool(row.get("strict_gate_pass")),
                "fitness_score": row.get("fitness_score"),
                "pareto_rank": row.get("pareto_rank"),
                "monthly_roi_pct": row.get("monthly_roi_pct"),
                "net_pnl": row.get("net_pnl"),
                "trade_count": row.get("trade_count"),
                "settled_count": row.get("settled_count"),
                "paper_days": row.get("paper_days"),
                "blockers": list(row.get("blockers") or []),
                "hard_vetoes": list(row.get("hard_vetoes") or []),
                "requested_targets": target_portfolios,
                "resolved_targets": [resolve_target_portfolio(target) for target in target_portfolios],
                "matched_targets": [
                    target for target in target_portfolios if resolve_target_portfolio(target) == str(portfolio_id)
                ],
                "execution_validation": dict(row.get("execution_validation") or {}),
                "package": _package_summary(package_path),
                "package_payload": package_payload,
                "artifact_payloads": artifact_payloads,
                "strategy_profile": _strategy_profile(
                    row,
                    genome_parameters=genome_parameters,
                    artifact_payloads=artifact_payloads,
                ),
            }
        )
    return sorted(refs, key=_candidate_context_sort_key)


def approve_manifest_for_live(manifest_id: str, *, approved_by: str, note: str | None = None) -> AcceptedStrategyManifest | None:
    return _registry().approve_manifest(manifest_id, approved_by=approved_by, note=note)


def reject_manifest(manifest_id: str, *, note: str | None = None) -> AcceptedStrategyManifest | None:
    return _registry().reject_manifest(manifest_id, note=note)
