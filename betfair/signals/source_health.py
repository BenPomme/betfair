from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceHealthState:
    source: str
    status: str = "idle"
    last_success_ts: Optional[str] = None
    last_error_ts: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None
    freshness_seconds: Optional[float] = None
    item_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SourceHealthTracker:
    def __init__(self) -> None:
        self._state: Dict[str, SourceHealthState] = {}

    def mark_success(self, source: str, *, item_count: int = 0, observed_ts: Optional[str] = None) -> None:
        state = self._state.setdefault(source, SourceHealthState(source=source))
        state.status = "healthy"
        state.last_success_ts = observed_ts or _utc_now_iso()
        state.success_count += 1
        state.item_count = int(item_count or 0)
        state.last_error = None
        state.freshness_seconds = 0.0

    def mark_error(self, source: str, error: Exception | str) -> None:
        state = self._state.setdefault(source, SourceHealthState(source=source))
        state.status = "degraded"
        state.last_error_ts = _utc_now_iso()
        state.error_count += 1
        state.last_error = str(error)

    def update_freshness(self, source: str, freshness_seconds: Optional[float]) -> None:
        state = self._state.setdefault(source, SourceHealthState(source=source))
        state.freshness_seconds = None if freshness_seconds is None else round(float(freshness_seconds), 2)
        if freshness_seconds is not None and freshness_seconds > 60 and state.status == "healthy":
            state.status = "stale"

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {source: state.to_dict() for source, state in sorted(self._state.items())}
