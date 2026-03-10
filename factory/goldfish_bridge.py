from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

from factory.contracts import AcceptedStrategyManifest, ExperimentSpec, ManifestStatus, PromotionStage, utc_now_iso


class GoldfishBridge:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def workspace_path(self, family_id: str) -> Path:
        return self.root / family_id

    def ensure_workspace(self, family_id: str, *, thesis: str, pipeline_stages: Iterable[str]) -> Dict[str, Any]:
        workspace = self.workspace_path(family_id)
        workspace.mkdir(parents=True, exist_ok=True)
        stage_names = list(pipeline_stages)
        state_path = workspace / "STATE.md"
        if not state_path.exists():
            state_path.write_text(
                "\n".join(
                    [
                        f"# {family_id}",
                        "",
                        "## Active Goal",
                        thesis,
                        "",
                        "## Recent Actions",
                        f"- [{utc_now_iso()}] Workspace initialized for research lineage packaging.",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        pipeline_path = workspace / "pipeline.yaml"
        if not pipeline_path.exists():
            lines = ["stages:"]
            previous = None
            for stage_name in stage_names:
                lines.append(f"  - name: {stage_name}")
                if previous is None:
                    lines.append("    inputs: {}")
                else:
                    lines.append(f"    inputs: {{upstream: {{from_stage: {previous}, signal: primary}}}}")
                lines.append("    outputs: {primary: {type: file}}")
                previous = stage_name
            pipeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "family_id": family_id,
            "workspace": str(workspace),
            "pipeline_path": str(pipeline_path),
            "state_path": str(state_path),
            "ready": state_path.exists() and pipeline_path.exists(),
        }

    def publish_candidate_manifest(
        self,
        *,
        lineage_id: str,
        family_id: str,
        portfolio_targets: List[str],
        venue_targets: List[str],
        artifact_refs: Dict[str, Any],
        runtime_overrides: Dict[str, Any],
        notes: List[str] | None = None,
    ) -> AcceptedStrategyManifest:
        checksum = hashlib.sha1(
            f"{lineage_id}:{family_id}:{sorted(portfolio_targets)}:{sorted(artifact_refs.keys())}".encode("utf-8")
        ).hexdigest()
        manifest_id = f"{family_id}-{checksum[:12]}"
        return AcceptedStrategyManifest(
            manifest_id=manifest_id,
            lineage_id=lineage_id,
            family_id=family_id,
            portfolio_targets=list(portfolio_targets),
            venue_targets=list(venue_targets),
            approved_stage=PromotionStage.LIVE_READY.value,
            status=ManifestStatus.PENDING_APPROVAL.value,
            artifact_refs=dict(artifact_refs),
            runtime_overrides=dict(runtime_overrides),
            checksum=checksum,
            notes=list(notes or []),
        )
