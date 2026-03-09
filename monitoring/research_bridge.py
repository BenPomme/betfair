from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import config


_FAMILY_BY_PORTFOLIO = {
    "betfair_core": "betfair_prediction",
    "betfair_prediction_league": "betfair_prediction",
    "betfair_suspension_lag": "betfair_info_arb",
    "betfair_crossbook_consensus": "betfair_info_arb",
    "betfair_timezone_decay": "betfair_info_arb",
    "polymarket_quantum_fold": "polymarket",
    "polymarket_binary_research": "polymarket",
    "hedge_validation": "funding",
    "hedge_research": "funding",
    "cascade_alpha": "funding",
    "contrarian_legacy": "funding",
}


def _runtime_root() -> Path:
    return Path(getattr(config, "RESEARCH_RUNTIME_ROOT", "data/research"))


def _manifest_dir() -> Path:
    path = _runtime_root() / "goldfish" / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def family_for_portfolio(portfolio_id: str) -> Optional[str]:
    return _FAMILY_BY_PORTFOLIO.get(str(portfolio_id or ""))


def latest_manifest_path(family: str) -> Path:
    return _manifest_dir() / f"{family}.latest.json"


def history_manifest_path(family: str) -> Path:
    return _manifest_dir() / f"{family}.history.jsonl"


def load_latest_research_run(portfolio_id: str) -> Optional[Dict[str, Any]]:
    family = family_for_portfolio(portfolio_id)
    if not family:
        return None
    path = latest_manifest_path(family)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def _mirror_to_legacy_logs(payload: Dict[str, Any]) -> None:
    family = str(payload.get("family") or "")
    base_row = {
        "run_id": payload.get("run_id"),
        "ts_start": payload.get("started_at"),
        "ts_end": payload.get("finished_at"),
        "git_sha": payload.get("git_sha"),
        "config_hash": payload.get("config_hash"),
        "params": payload.get("params") or {},
        "metrics": payload.get("metrics") or {},
        "gate": payload.get("gate") or {},
        "research": {
            "family": family,
            "status": payload.get("status"),
            "decision": payload.get("decision"),
            "artifact_manifest": payload.get("artifact_manifest"),
        },
    }
    if family == "betfair_prediction":
        row = {
            **base_row,
            "model_id": payload.get("subject") or family,
            "model_kind": "goldfish_reseed",
            "models": payload.get("models") or {},
        }
        _append_jsonl(Path(config.PREDICTION_EXPERIMENT_LOG_PATH), row)
    elif family == "funding":
        row = {
            **base_row,
            "model_family": payload.get("subject") or family,
            "models": payload.get("models") or {},
            "mode": "goldfish",
        }
        _append_jsonl(Path(config.FUNDING_EXPERIMENT_LOG_PATH), row)


def publish_research_run(
    *,
    family: str,
    run_id: str,
    status: str,
    decision: str,
    portfolios: list[str],
    subject: str,
    git_sha: str,
    metrics: Optional[Dict[str, Any]] = None,
    gate: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    models: Optional[Dict[str, Any]] = None,
    artifact_manifest: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "family": family,
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "portfolios": list(portfolios),
        "subject": subject,
        "git_sha": git_sha,
        "metrics": metrics or {},
        "gate": gate or {},
        "params": params or {},
        "models": models or {},
        "artifact_manifest": artifact_manifest or {},
        "started_at": started_at,
        "finished_at": finished_at,
        "config_hash": config_hash,
    }
    latest_manifest_path(family).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_jsonl(history_manifest_path(family), payload)
    if str(status).lower() in {"finalized", "accepted", "published"}:
        _mirror_to_legacy_logs(payload)
    return payload
