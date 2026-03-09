from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _family_sources(family: str) -> Dict[str, Path]:
    return {
        "betfair_prediction": DATA_ROOT / "prediction" / "experiments.jsonl",
        "betfair_info_arb": DATA_ROOT / "portfolios" / "betfair_core" / "state.json",
        "polymarket": DATA_ROOT / "portfolios" / "polymarket_quantum_fold" / "state.json",
        "funding": DATA_ROOT / "funding" / "experiments.jsonl",
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


def _dataset_payload(family: str) -> Dict[str, Any]:
    source = _family_sources(family)
    path = source.get(family)
    if path is None:
        return {"family": family, "rows": 0, "source": None}
    if path.suffix == ".json":
        payload = _read_json(path, {})
        return {"family": family, "rows": 1 if payload else 0, "source": str(path), "payload": payload}
    rows = _read_jsonl(path)
    return {"family": family, "rows": len(rows), "source": str(path), "payload": rows}


def _stage_payload(family: str, stage: str) -> Dict[str, Any]:
    dataset = _dataset_payload(family)
    if stage == "extract_dataset":
        return {"stage": stage, "git_sha": _git_sha(), **dataset}
    if stage == "build_features":
        return {"stage": stage, "git_sha": _git_sha(), "family": family, "feature_rows": dataset.get("rows", 0)}
    if stage == "train_candidates":
        return {"stage": stage, "git_sha": _git_sha(), "family": family, "candidate_models": dataset.get("rows", 0)}
    if stage == "replay_or_walkforward":
        return {"stage": stage, "git_sha": _git_sha(), "family": family, "replay_rows": dataset.get("rows", 0)}
    if stage == "calibrate":
        return {"stage": stage, "git_sha": _git_sha(), "family": family, "calibration_rows": dataset.get("rows", 0)}
    if stage == "publish_artifact_manifest":
        return {
            "stage": stage,
            "git_sha": _git_sha(),
            "family": family,
            "artifact_manifest": {
                "family": family,
                "published_from_git_sha": _git_sha(),
                "source_rows": dataset.get("rows", 0),
            },
        }
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = _stage_payload(args.family, args.stage)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
