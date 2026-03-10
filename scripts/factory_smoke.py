#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env", override=True)
except ImportError:
    pass

import config  # noqa: E402
from factory.orchestrator import FactoryOrchestrator  # noqa: E402


def _summary(state: dict) -> dict:
    lineages = list(state.get("lineages") or [])
    families = list(state.get("families") or [])
    manifests = dict(state.get("manifests") or {})
    return {
        "agentic_factory_mode": state.get("agentic_factory_mode"),
        "agentic_tokens_allowed": state.get("agentic_tokens_allowed"),
        "factory_influence_allowed": state.get("factory_influence_allowed"),
        "cycle_count": state.get("cycle_count"),
        "status": state.get("status"),
        "readiness": dict(state.get("readiness") or {}),
        "research_summary": dict(state.get("research_summary") or {}),
        "learning_memory_count": int(((state.get("research_summary") or {}).get("learning_memory_count", 0)) or 0),
        "family_count": len(families),
        "lineage_count": len(lineages),
        "artifact_backed_lineages": [
            {
                "lineage_id": row.get("lineage_id"),
                "family_id": row.get("family_id"),
                "role": row.get("role"),
                "current_stage": row.get("current_stage"),
                "package": row.get("latest_artifact_package"),
                "mode": row.get("latest_artifact_mode"),
            }
            for row in lineages
            if row.get("latest_artifact_package")
        ],
        "top_lineages": sorted(
            [
                {
                    "lineage_id": row.get("lineage_id"),
                    "family_id": row.get("family_id"),
                    "role": row.get("role"),
                    "current_stage": row.get("current_stage"),
                    "monthly_roi_pct": row.get("monthly_roi_pct"),
                    "fitness_score": row.get("fitness_score"),
                    "pareto_rank": row.get("pareto_rank"),
                }
                for row in lineages
            ],
            key=lambda item: (
                item.get("pareto_rank") if item.get("pareto_rank") is not None else 999,
                -float(item.get("fitness_score", 0.0) or 0.0),
            ),
        )[:10],
        "pending_manifest_count": len(manifests.get("pending") or []),
        "live_manifest_count": len(manifests.get("live_loadable") or []),
    }


def _print_human(summary: dict) -> None:
    print(f"mode={summary['agentic_factory_mode']} status={summary['status']} cycles={summary.get('cycle_count')}")
    print(
        "families={family_count} lineages={lineage_count} artifact_backed={artifact_count} pending_manifests={pending} live_manifests={live}".format(
            family_count=summary["family_count"],
            lineage_count=summary["lineage_count"],
            artifact_count=len(summary["artifact_backed_lineages"]),
            pending=summary["pending_manifest_count"],
            live=summary["live_manifest_count"],
        )
    )
    print(f"learning_memory={summary['learning_memory_count']}")
    readiness = summary.get("readiness") or {}
    print(
        "readiness={status} blockers={blockers}".format(
            status=readiness.get("status"),
            blockers=",".join(readiness.get("blockers") or []) or "none",
        )
    )
    print("top_lineages:")
    for row in summary["top_lineages"]:
        print(
            "  - {family_id} {lineage_id} {current_stage} roi={roi} fitness={fitness} rank={rank}".format(
                family_id=row.get("family_id"),
                lineage_id=row.get("lineage_id"),
                current_stage=row.get("current_stage"),
                roi=row.get("monthly_roi_pct"),
                fitness=row.get("fitness_score"),
                rank=row.get("pareto_rank"),
            )
        )
    if summary["artifact_backed_lineages"]:
        print("artifact_backed_lineages:")
        for row in summary["artifact_backed_lineages"][:10]:
            print(
                "  - {family_id} {lineage_id} {mode} package={package}".format(
                    family_id=row.get("family_id"),
                    lineage_id=row.get("lineage_id"),
                    mode=row.get("mode"),
                    package=row.get("package"),
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local research-factory smoke cycle and print a compact summary.")
    parser.add_argument("--project-root", default=str(project_root), help="Project root to evaluate.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of orchestrator cycles to execute.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human summary.")
    parser.add_argument("--factory-root", help="Override FACTORY_ROOT for this smoke run.")
    parser.add_argument("--goldfish-root", help="Override FACTORY_GOLDFISH_ROOT for this smoke run.")
    parser.add_argument("--portfolio-root", help="Override PORTFOLIO_STATE_ROOT for this smoke run.")
    args = parser.parse_args(argv)

    if args.factory_root:
        setattr(config, "FACTORY_ROOT", args.factory_root)
    if args.goldfish_root:
        setattr(config, "FACTORY_GOLDFISH_ROOT", args.goldfish_root)
    if args.portfolio_root:
        setattr(config, "PORTFOLIO_STATE_ROOT", args.portfolio_root)

    orchestrator = FactoryOrchestrator(Path(args.project_root))
    state = {}
    for _ in range(max(1, int(args.cycles))):
        state = orchestrator.run_cycle()
    summary = _summary(state)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
