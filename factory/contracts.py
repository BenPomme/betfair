from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonMixin:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromotionStage(str, Enum):
    IDEA = "idea"
    SPEC = "spec"
    DATA_CHECK = "data_check"
    GOLDFISH_RUN = "goldfish_run"
    WALKFORWARD = "walkforward"
    STRESS = "stress"
    SHADOW = "shadow"
    PAPER = "paper"
    CANARY_READY = "canary_ready"
    LIVE_READY = "live_ready"
    APPROVED_LIVE = "approved_live"


class LineageRole(str, Enum):
    CHAMPION = "champion"
    SHADOW_CHALLENGER = "shadow_challenger"
    PAPER_CHALLENGER = "paper_challenger"
    MOONSHOT = "moonshot"


class EvaluationStage(str, Enum):
    WALKFORWARD = "walkforward"
    STRESS = "stress"
    SHADOW = "shadow"
    PAPER = "paper"


class ManifestStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED_LIVE = "approved_live"
    REJECTED = "rejected"


class ExecutionTier(str, Enum):
    TIER0 = "tier0_deterministic"
    TIER1 = "tier1_local_cheap"
    TIER2 = "tier2_local_strong"
    TIER3 = "tier3_review_gate"


@dataclass
class ConnectorSnapshot(JsonMixin):
    connector_id: str
    venue: str
    data_products: List[str]
    ready: bool
    latest_data_ts: Optional[str] = None
    record_count: int = 0
    source_paths: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@runtime_checkable
class ConnectorAdapter(Protocol):
    connector_id: str
    venue: str

    def snapshot(self) -> ConnectorSnapshot:
        ...


@dataclass
class ResearchHypothesis(JsonMixin):
    hypothesis_id: str
    family_id: str
    title: str
    thesis: str
    scientific_domains: List[str]
    lead_agent_role: str
    success_metric: str
    guardrails: List[str]
    collaborating_agent_roles: List[str] = field(default_factory=list)
    origin: str = "seeded_family"
    agent_notes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class MutationBounds(JsonMixin):
    horizons_seconds: List[int] = field(default_factory=list)
    feature_subsets: List[str] = field(default_factory=list)
    model_classes: List[str] = field(default_factory=list)
    execution_thresholds: Dict[str, List[float]] = field(default_factory=dict)
    hyperparameter_ranges: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class StrategyGenome(JsonMixin):
    genome_id: str
    lineage_id: str
    family_id: str
    parent_genome_id: Optional[str]
    role: str
    parameters: Dict[str, Any]
    mutation_bounds: MutationBounds
    scientific_domains: List[str]
    budget_bucket: str
    resource_profile: str
    budget_weight_pct: float
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class ExperimentSpec(JsonMixin):
    experiment_id: str
    lineage_id: str
    family_id: str
    hypothesis_id: str
    genome_id: str
    goldfish_workspace: str
    pipeline_stages: List[str]
    backend_mode: str
    resource_profile: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class ExperimentQueueEntry(JsonMixin):
    queue_id: str
    family_id: str
    lineage_id: str
    experiment_id: str
    role: str
    current_stage: str
    status: str
    priority: int
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    notes: List[str] = field(default_factory=list)


@dataclass
class EvaluationWindow(JsonMixin):
    label: str
    settled_count: int
    monthly_roi_pct: float
    baseline_roi_pct: float
    brier_lift_abs: float
    drawdown_pct: float
    slippage_headroom_pct: float
    failure_rate: float
    regime_robustness: float


@dataclass
class EvaluationBundle(JsonMixin):
    evaluation_id: str
    lineage_id: str
    family_id: str
    stage: str
    source: str
    generated_at: str = field(default_factory=utc_now_iso)
    windows: List[EvaluationWindow] = field(default_factory=list)
    monthly_roi_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    slippage_headroom_pct: float = 0.0
    calibration_lift_abs: float = 0.0
    turnover: float = 0.0
    capacity_score: float = 0.0
    failure_rate: float = 0.0
    regime_robustness: float = 0.0
    baseline_beaten_windows: int = 0
    stress_positive: bool = False
    trade_count: int = 0
    settled_count: int = 0
    paper_days: int = 0
    net_pnl: float = 0.0
    hard_vetoes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    pareto_rank: Optional[int] = None
    fitness_score: Optional[float] = None


@dataclass
class PromotionDecision(JsonMixin):
    lineage_id: str
    current_stage: str
    next_stage: str
    allowed: bool
    requires_human_signoff: bool
    blockers: List[str]
    reasons: List[str]
    evidence_ids: List[str] = field(default_factory=list)
    decided_at: str = field(default_factory=utc_now_iso)


@dataclass
class LearningMemoryEntry(JsonMixin):
    memory_id: str
    family_id: str
    lineage_id: str
    hypothesis_id: str
    outcome: str
    summary: str
    scientific_domains: List[str]
    lead_agent_role: str
    tweak_count: int
    decision_stage: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class AcceptedStrategyManifest(JsonMixin):
    manifest_id: str
    lineage_id: str
    family_id: str
    portfolio_targets: List[str]
    venue_targets: List[str]
    approved_stage: str
    status: str
    artifact_refs: Dict[str, Any]
    runtime_overrides: Dict[str, Any]
    created_at: str = field(default_factory=utc_now_iso)
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    checksum: str = ""
    notes: List[str] = field(default_factory=list)

    def is_live_loadable(self) -> bool:
        return (
            self.status == ManifestStatus.APPROVED_LIVE.value
            and self.approved_stage == PromotionStage.APPROVED_LIVE.value
            and bool(self.approved_by)
            and bool(self.approved_at)
        )


@dataclass
class LineageRecord(JsonMixin):
    lineage_id: str
    family_id: str
    label: str
    role: str
    current_stage: str
    target_portfolios: List[str]
    target_venues: List[str]
    hypothesis_id: str
    genome_id: str
    experiment_id: str
    budget_bucket: str
    budget_weight_pct: float
    connector_ids: List[str]
    goldfish_workspace: str
    revision: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_decision: Optional[Dict[str, Any]] = None
    last_evaluation_id: Optional[str] = None
    last_manifest_id: Optional[str] = None
    blockers: List[str] = field(default_factory=list)
    active: bool = True
    loss_streak: int = 0
    tweak_count: int = 0
    max_tweaks: int = 2
    iteration_status: str = "new_candidate"
    parent_lineage_id: Optional[str] = None
    retired_at: Optional[str] = None
    retirement_reason: Optional[str] = None
    last_memory_id: Optional[str] = None


@dataclass
class FactoryFamily(JsonMixin):
    family_id: str
    label: str
    thesis: str
    target_portfolios: List[str]
    target_venues: List[str]
    primary_connector_ids: List[str]
    champion_lineage_id: str
    shadow_challenger_ids: List[str]
    paper_challenger_ids: List[str]
    budget_split: Dict[str, float]
    queue_stage: str
    explainer: str
    last_cycle_at: Optional[str] = None
    retired_lineage_ids: List[str] = field(default_factory=list)


@dataclass
class FactoryJournal(JsonMixin):
    active_goal: str
    recent_actions: List[str]
    updated_at: str = field(default_factory=utc_now_iso)
