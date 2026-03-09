from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from monitoring.research_bridge import publish_research_run


ROOT = Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _loads(value: str):
    if not value:
        return {}
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--portfolios", required=True)
    parser.add_argument("--metrics", default="{}")
    parser.add_argument("--gate", default="{}")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--models", default="{}")
    parser.add_argument("--artifact-manifest", default="{}")
    parser.add_argument("--started-at", default=None)
    parser.add_argument("--finished-at", default=None)
    parser.add_argument("--config-hash", default=None)
    args = parser.parse_args()
    payload = publish_research_run(
        family=args.family,
        run_id=args.run_id,
        status=args.status,
        decision=args.decision,
        portfolios=[item.strip() for item in args.portfolios.split(",") if item.strip()],
        subject=args.subject,
        git_sha=_git_sha(),
        metrics=_loads(args.metrics),
        gate=_loads(args.gate),
        params=_loads(args.params),
        models=_loads(args.models),
        artifact_manifest=_loads(args.artifact_manifest),
        started_at=args.started_at,
        finished_at=args.finished_at,
        config_hash=args.config_hash,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
