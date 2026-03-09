from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

PortfolioControlMode = Literal["local_managed", "disabled"]


@dataclass
class StrategyAccount:
    portfolio_id: str
    currency: str
    starting_balance: float
    current_balance: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    slippage_cost: float = 0.0
    gross_exposure: float = 0.0
    roi_pct: float = 0.0
    drawdown_pct: float = 0.0
    wins: int = 0
    losses: int = 0
    trade_count: int = 0
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["StrategyAccount"]:
        if not data:
            return None
        return cls(**data)


@dataclass
class ModelShadowAccount:
    portfolio_id: str
    model_id: str
    shadow_starting_balance: float
    shadow_current_balance: float
    shadow_realized_pnl: float = 0.0
    shadow_roi_pct: float = 0.0
    settled_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    selected_for_execution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ModelShadowAccount"]:
        if not data:
            return None
        return cls(**data)


@dataclass
class PortfolioSummary:
    portfolio_id: str
    label: str
    category: str
    control_mode: PortfolioControlMode
    running: bool
    mode: str
    bankroll: float
    currency: str
    realized_pnl: float
    unrealized_pnl: float
    roi_pct: float
    max_drawdown_pct: float
    open_count: int
    readiness: str
    last_heartbeat_ts: Optional[str]
    progress_pct: float = 0.0
    trend_direction: str = "flat"
    progress_delta_24h: float = 0.0
    blocker_count: int = 0
    eta_to_readiness: Optional[str] = None
    eta_hours: Optional[float] = None
    status: str = "idle"
    process_pid: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    audit_state: str = "learning_ok"
    audit_owner: str = "ops"
    audit_next_action: Optional[str] = None
    last_progress_report_ts: Optional[str] = None
    latest_research_run: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioState:
    portfolio_id: str
    running: bool
    read_only: bool
    status: str
    account: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    execution_quality: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    models: List[Dict[str, Any]] = field(default_factory=list)
    balance_history: List[Dict[str, Any]] = field(default_factory=list)
    raw_state: Dict[str, Any] = field(default_factory=dict)
    control_mode: PortfolioControlMode = "local_managed"
    error: Optional[str] = None
    trend: Dict[str, Any] = field(default_factory=dict)
    audit_state: str = "learning_ok"
    audit_owner: str = "ops"
    audit_next_action: Optional[str] = None
    last_progress_report_ts: Optional[str] = None
    latest_research_run: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioRunnerSpec:
    portfolio_id: str
    label: str
    category: str
    control_mode: PortfolioControlMode
    currency: str
    initial_balance: float
    runner_path: str = ""
    autostart: bool = False
    enabled: bool = True
    description: str = ""
    ui_group: str = "Portfolios"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
