from __future__ import annotations

import json
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, TypeVar

from factory.contracts import (
    AcceptedStrategyManifest,
    EvaluationBundle,
    EvaluationWindow,
    ExperimentSpec,
    FactoryFamily,
    FactoryJournal,
    LearningMemoryEntry,
    LineageRecord,
    ManifestStatus,
    MutationBounds,
    PromotionStage,
    ResearchHypothesis,
    StrategyGenome,
    utc_now_iso,
)

T = TypeVar("T")


def _coerce_payload(cls: Type[T], payload: Dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    clean = {key: value for key, value in payload.items() if key in allowed}
    if cls is StrategyGenome and isinstance(clean.get("mutation_bounds"), dict):
        clean["mutation_bounds"] = _coerce_payload(MutationBounds, clean["mutation_bounds"])
    if cls is EvaluationBundle:
        clean["windows"] = [
            _coerce_payload(EvaluationWindow, item)
            for item in (clean.get("windows") or [])
            if isinstance(item, dict)
        ]
    return cls(**clean)


class FactoryRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.catalog_path = self.root / "catalog.json"
        self.families_dir = self.root / "families"
        self.lineages_dir = self.root / "lineages"
        self.evaluations_dir = self.root / "evaluations"
        self.manifests_dir = self.root / "manifests"
        self.history_dir = self.root / "history"
        self.state_dir = self.root / "state"
        self.journal_path = self.state_dir / "STATE.md"
        self._lock = threading.Lock()
        self.ensure()

    def ensure(self) -> None:
        for path in [
            self.root,
            self.families_dir,
            self.lineages_dir,
            self.evaluations_dir,
            self.manifests_dir,
            self.history_dir,
            self.state_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        if not self.catalog_path.exists():
            self._atomic_write_json(
                self.catalog_path,
                {"created_at": utc_now_iso(), "families": [], "lineages": [], "manifests": []},
            )

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def _read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def _catalog(self) -> Dict[str, Any]:
        return self._read_json(self.catalog_path, default={}) or {}

    def _write_catalog(self, payload: Dict[str, Any]) -> None:
        self._atomic_write_json(self.catalog_path, payload)

    def save_family(self, family: FactoryFamily) -> None:
        with self._lock:
            self._atomic_write_json(self.families_dir / f"{family.family_id}.json", family.to_dict())
            catalog = self._catalog()
            families = set(catalog.get("families") or [])
            families.add(family.family_id)
            catalog["families"] = sorted(families)
            self._write_catalog(catalog)

    def families(self) -> List[FactoryFamily]:
        return [
            _coerce_payload(FactoryFamily, payload)
            for payload in (
                self._read_json(path, default={})
                for path in sorted(self.families_dir.glob("*.json"))
            )
            if payload
        ]

    def load_family(self, family_id: str) -> Optional[FactoryFamily]:
        payload = self._read_json(self.families_dir / f"{family_id}.json", default={})
        if not payload:
            return None
        return _coerce_payload(FactoryFamily, payload)

    def save_research_pack(
        self,
        *,
        hypothesis: ResearchHypothesis,
        genome: StrategyGenome,
        experiment: ExperimentSpec,
        lineage: LineageRecord,
    ) -> None:
        with self._lock:
            lineage_dir = self.lineages_dir / lineage.lineage_id
            lineage_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(lineage_dir / "hypothesis.json", hypothesis.to_dict())
            self._atomic_write_json(lineage_dir / "genome.json", genome.to_dict())
            self._atomic_write_json(lineage_dir / "experiment.json", experiment.to_dict())
            self._atomic_write_json(lineage_dir / "lineage.json", lineage.to_dict())
            catalog = self._catalog()
            lineages = set(catalog.get("lineages") or [])
            lineages.add(lineage.lineage_id)
            catalog["lineages"] = sorted(lineages)
            self._write_catalog(catalog)

    def lineages(self) -> List[LineageRecord]:
        records: List[LineageRecord] = []
        for path in sorted(self.lineages_dir.glob("*/lineage.json")):
            payload = self._read_json(path, default={})
            if payload:
                records.append(_coerce_payload(LineageRecord, payload))
        return records

    def load_lineage(self, lineage_id: str) -> Optional[LineageRecord]:
        payload = self._read_json(self.lineages_dir / lineage_id / "lineage.json", default={})
        if not payload:
            return None
        return _coerce_payload(LineageRecord, payload)

    def load_hypothesis(self, lineage_id: str) -> Optional[ResearchHypothesis]:
        payload = self._read_json(self.lineages_dir / lineage_id / "hypothesis.json", default={})
        if not payload:
            return None
        return _coerce_payload(ResearchHypothesis, payload)

    def load_genome(self, lineage_id: str) -> Optional[StrategyGenome]:
        payload = self._read_json(self.lineages_dir / lineage_id / "genome.json", default={})
        if not payload:
            return None
        return _coerce_payload(StrategyGenome, payload)

    def load_experiment(self, lineage_id: str) -> Optional[ExperimentSpec]:
        payload = self._read_json(self.lineages_dir / lineage_id / "experiment.json", default={})
        if not payload:
            return None
        return _coerce_payload(ExperimentSpec, payload)

    def save_lineage(self, lineage: LineageRecord) -> None:
        with self._lock:
            self._atomic_write_json(self.lineages_dir / lineage.lineage_id / "lineage.json", lineage.to_dict())

    def save_genome(self, lineage_id: str, genome: StrategyGenome) -> None:
        with self._lock:
            self._atomic_write_json(self.lineages_dir / lineage_id / "genome.json", genome.to_dict())

    def save_hypothesis(self, lineage_id: str, hypothesis: ResearchHypothesis) -> None:
        with self._lock:
            self._atomic_write_json(self.lineages_dir / lineage_id / "hypothesis.json", hypothesis.to_dict())

    def save_experiment(self, lineage_id: str, experiment: ExperimentSpec) -> None:
        with self._lock:
            self._atomic_write_json(self.lineages_dir / lineage_id / "experiment.json", experiment.to_dict())

    def cas_transition(
        self,
        lineage_id: str,
        *,
        expected_stage: str,
        next_stage: str,
        blockers: Sequence[str],
        decision: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            lineage = self.load_lineage(lineage_id)
            if lineage is None or lineage.current_stage != expected_stage:
                return False
            lineage.current_stage = next_stage
            lineage.blockers = list(blockers)
            lineage.last_decision = dict(decision or {})
            lineage.revision += 1
            lineage.updated_at = utc_now_iso()
            self._atomic_write_json(
                self.lineages_dir / lineage_id / "lineage.json",
                lineage.to_dict(),
            )
            self._append_jsonl(
                self.history_dir / "promotions.jsonl",
                {
                    "ts": lineage.updated_at,
                    "lineage_id": lineage.lineage_id,
                    "expected_stage": expected_stage,
                    "next_stage": next_stage,
                    "revision": lineage.revision,
                    "blockers": list(blockers),
                    "decision": dict(decision or {}),
                },
            )
            return True

    def save_evaluation(self, bundle: EvaluationBundle) -> None:
        with self._lock:
            lineage_dir = self.evaluations_dir / bundle.lineage_id
            lineage_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(lineage_dir / f"{bundle.evaluation_id}.json", bundle.to_dict())
            self._append_jsonl(
                self.history_dir / "fitness_scores.jsonl",
                {
                    "ts": bundle.generated_at,
                    "lineage_id": bundle.lineage_id,
                    "evaluation_id": bundle.evaluation_id,
                    "stage": bundle.stage,
                    "pareto_rank": bundle.pareto_rank,
                    "fitness_score": bundle.fitness_score,
                    "monthly_roi_pct": bundle.monthly_roi_pct,
                    "max_drawdown_pct": bundle.max_drawdown_pct,
                    "hard_vetoes": list(bundle.hard_vetoes),
                },
            )
            if bundle.stage == "paper":
                self._append_jsonl(
                    self.history_dir / "paper_league.jsonl",
                    {
                        "ts": bundle.generated_at,
                        "lineage_id": bundle.lineage_id,
                        "family_id": bundle.family_id,
                        "monthly_roi_pct": bundle.monthly_roi_pct,
                        "trade_count": bundle.trade_count,
                        "paper_days": bundle.paper_days,
                    },
                )
            lineage = self.load_lineage(bundle.lineage_id)
            if lineage is not None:
                lineage.last_evaluation_id = bundle.evaluation_id
                lineage.updated_at = utc_now_iso()
                self._atomic_write_json(self.lineages_dir / bundle.lineage_id / "lineage.json", lineage.to_dict())

    def save_learning_memory(self, memory: LearningMemoryEntry) -> None:
        with self._lock:
            self._append_jsonl(self.history_dir / "learning_memory.jsonl", memory.to_dict())

    def learning_memories(
        self,
        *,
        family_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LearningMemoryEntry]:
        rows = self._read_jsonl(self.history_dir / "learning_memory.jsonl")
        memories = [
            _coerce_payload(LearningMemoryEntry, row)
            for row in rows
            if row and (family_id is None or row.get("family_id") == family_id)
        ]
        if limit is not None and limit >= 0:
            return memories[-limit:]
        return memories

    def evaluations(self, lineage_id: Optional[str] = None) -> List[EvaluationBundle]:
        paths = sorted((self.evaluations_dir / lineage_id).glob("*.json")) if lineage_id else sorted(self.evaluations_dir.glob("*/*.json"))
        bundles: List[EvaluationBundle] = []
        for path in paths:
            payload = self._read_json(path, default={})
            if payload:
                bundles.append(_coerce_payload(EvaluationBundle, payload))
        return bundles

    def latest_evaluation_by_stage(self, lineage_id: str) -> Dict[str, EvaluationBundle]:
        latest: Dict[str, EvaluationBundle] = {}
        for bundle in self.evaluations(lineage_id=lineage_id):
            previous = latest.get(bundle.stage)
            if previous is None or str(bundle.generated_at) > str(previous.generated_at):
                latest[bundle.stage] = bundle
        return latest

    def save_manifest(self, manifest: AcceptedStrategyManifest) -> None:
        with self._lock:
            self._atomic_write_json(self.manifests_dir / f"{manifest.manifest_id}.json", manifest.to_dict())
            catalog = self._catalog()
            manifests = set(catalog.get("manifests") or [])
            manifests.add(manifest.manifest_id)
            catalog["manifests"] = sorted(manifests)
            self._write_catalog(catalog)
            self._append_jsonl(
                self.history_dir / "manifests.jsonl",
                {
                    "ts": utc_now_iso(),
                    "manifest_id": manifest.manifest_id,
                    "lineage_id": manifest.lineage_id,
                    "family_id": manifest.family_id,
                    "status": manifest.status,
                    "approved_stage": manifest.approved_stage,
                },
            )
            lineage = self.load_lineage(manifest.lineage_id)
            if lineage is not None:
                lineage.last_manifest_id = manifest.manifest_id
                lineage.updated_at = utc_now_iso()
                self._atomic_write_json(self.lineages_dir / manifest.lineage_id / "lineage.json", lineage.to_dict())

    def manifests(self) -> List[AcceptedStrategyManifest]:
        manifests: List[AcceptedStrategyManifest] = []
        for path in sorted(self.manifests_dir.glob("*.json")):
            payload = self._read_json(path, default={})
            if payload:
                manifests.append(_coerce_payload(AcceptedStrategyManifest, payload))
        return manifests

    def load_manifest(self, manifest_id: str) -> Optional[AcceptedStrategyManifest]:
        payload = self._read_json(self.manifests_dir / f"{manifest_id}.json", default={})
        if not payload:
            return None
        return _coerce_payload(AcceptedStrategyManifest, payload)

    def approve_manifest(
        self,
        manifest_id: str,
        *,
        approved_by: str,
        note: Optional[str] = None,
    ) -> Optional[AcceptedStrategyManifest]:
        with self._lock:
            manifest = self.load_manifest(manifest_id)
            if manifest is None:
                return None
            manifest.status = ManifestStatus.APPROVED_LIVE.value
            manifest.approved_stage = PromotionStage.APPROVED_LIVE.value
            manifest.approved_by = approved_by
            manifest.approved_at = utc_now_iso()
            if note:
                manifest.notes.append(note)
            self._atomic_write_json(self.manifests_dir / f"{manifest.manifest_id}.json", manifest.to_dict())
            self._append_jsonl(
                self.history_dir / "manifests.jsonl",
                {
                    "ts": manifest.approved_at,
                    "manifest_id": manifest.manifest_id,
                    "lineage_id": manifest.lineage_id,
                    "family_id": manifest.family_id,
                    "status": manifest.status,
                    "approved_stage": manifest.approved_stage,
                    "approved_by": approved_by,
                    "note": note,
                },
            )
            return manifest

    def reject_manifest(self, manifest_id: str, *, note: Optional[str] = None) -> Optional[AcceptedStrategyManifest]:
        with self._lock:
            manifest = self.load_manifest(manifest_id)
            if manifest is None:
                return None
            manifest.status = ManifestStatus.REJECTED.value
            if note:
                manifest.notes.append(note)
            self._atomic_write_json(self.manifests_dir / f"{manifest.manifest_id}.json", manifest.to_dict())
            self._append_jsonl(
                self.history_dir / "manifests.jsonl",
                {
                    "ts": utc_now_iso(),
                    "manifest_id": manifest.manifest_id,
                    "lineage_id": manifest.lineage_id,
                    "family_id": manifest.family_id,
                    "status": manifest.status,
                    "approved_stage": manifest.approved_stage,
                    "note": note,
                },
            )
            return manifest

    def live_manifests_for_portfolio(self, portfolio_id: str) -> List[AcceptedStrategyManifest]:
        return [
            manifest
            for manifest in self.manifests()
            if portfolio_id in manifest.portfolio_targets and manifest.is_live_loadable()
        ]

    def write_state(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write_json(self.state_dir / "summary.json", payload)

    def read_state(self) -> Dict[str, Any]:
        return self._read_json(self.state_dir / "summary.json", default={}) or {}

    def write_journal(self, journal: FactoryJournal) -> None:
        lines = ["# Research Factory", "", "## Active Goal", journal.active_goal, "", "## Recent Actions"]
        for action in journal.recent_actions:
            lines.append(f"- {action}")
        lines.append("")
        lines.append(f"_Updated: {journal.updated_at}_")
        self.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read_journal(self) -> FactoryJournal:
        if not self.journal_path.exists():
            return FactoryJournal(active_goal="Establish a reproducible strategy factory.", recent_actions=[])
        recent_actions: List[str] = []
        active_goal = "Establish a reproducible strategy factory."
        section = ""
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                section = line.strip()
                continue
            if section == "## Active Goal" and line.strip():
                active_goal = line.strip()
            if section == "## Recent Actions" and line.startswith("- "):
                recent_actions.append(line[2:])
        return FactoryJournal(active_goal=active_goal, recent_actions=recent_actions[-20:])
