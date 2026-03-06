"""
Position manager: track open hedge positions with JSON persistence.
In-memory cache + file-based state (no DB dependency for Phase 1).
"""
import json
import logging
import shutil
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import config
from funding.core.schemas import HedgePosition, HedgeStatus

logger = logging.getLogger(__name__)


class PositionManager:
    """Track open and closed hedge positions."""

    def __init__(self, state_path: Optional[str] = None):
        self._state_path = Path(state_path or config.FUNDING_STATE_PATH)
        self._positions: Dict[str, HedgePosition] = {}
        self._validation_run_id: str = ""
        self._fresh_book_started_at: Optional[str] = None
        self._archived_state_path: Optional[str] = None
        self._validation_dir: Optional[Path] = None
        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._validation_run_id = str(raw.get("validation_run_id", ""))
            self._fresh_book_started_at = raw.get("fresh_book_started_at")
            self._archived_state_path = raw.get("archived_state_path")
            if self._archived_state_path:
                self._validation_dir = Path(self._archived_state_path).parent
            for pos_data in raw.get("positions", []):
                pos = HedgePosition.from_dict(pos_data)
                self._positions[pos.symbol] = pos
            logger.info("Loaded %d positions from state", len(self._positions))
        except Exception as e:
            logger.warning("Failed to load position state: %s", e)

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "positions": [p.to_dict() for p in self._positions.values()],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "validation_run_id": self._validation_run_id,
            "fresh_book_started_at": self._fresh_book_started_at,
            "archived_state_path": self._archived_state_path,
        }
        self._state_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def archive_current_state(
        self,
        run_id: str,
        manifest: Optional[dict] = None,
    ) -> Optional[Path]:
        """Archive the current hedge state into a validation-run directory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = Path(config.FUNDING_VALIDATION_ARCHIVE_DIR) / f"{timestamp}_{run_id}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_state_path = archive_dir / self._state_path.name

        if self._state_path.exists():
            shutil.copy2(self._state_path, archived_state_path)
        else:
            archived_state_path.write_text(
                json.dumps({"positions": [], "last_updated": datetime.now(timezone.utc).isoformat()}, indent=2),
                encoding="utf-8",
            )

        manifest_payload = {
            "run_id": run_id,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "previous_position_count": len(self._positions),
        }
        if manifest:
            manifest_payload.update(manifest)
        (archive_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )
        return archived_state_path

    def initialize_fresh_validation_book(
        self,
        run_id: str,
        archived_state_path: Optional[Path] = None,
    ) -> None:
        """Reset hedge state for a fresh validation run."""
        self._positions = {}
        self._validation_run_id = run_id
        self._fresh_book_started_at = datetime.now(timezone.utc).isoformat()
        self._archived_state_path = str(archived_state_path) if archived_state_path else None
        self._validation_dir = archived_state_path.parent if archived_state_path else None
        self._save_state()

    def begin_validation_run(self, run_id: str, manifest: Optional[dict] = None) -> Optional[Path]:
        """Archive current hedge state and start a fresh validation book."""
        archived_state_path = self.archive_current_state(run_id, manifest=manifest)
        self.initialize_fresh_validation_book(run_id, archived_state_path=archived_state_path)
        return archived_state_path

    def log_rejection(self, reason: str, symbol: str, details: Optional[dict] = None) -> None:
        """Persist a rejected trade attempt to the current validation run."""
        if self._validation_dir is None:
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self._validation_run_id,
            "symbol": symbol,
            "reason": reason,
            "details": details or {},
        }
        with (self._validation_dir / "rejections.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def log_settlement_audit(self, event: dict) -> None:
        """Persist settlement audit data for the current validation run."""
        if self._validation_dir is None:
            return
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self._validation_run_id}
        payload.update(event)
        with (self._validation_dir / "settlement_audit.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def _read_jsonl(self, path: Path) -> List[dict]:
        if not path.exists():
            return []
        results: List[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                continue
        return results

    def get_recent_rejections(self, limit: int = 50) -> List[dict]:
        if self._validation_dir is None:
            return []
        return self._read_jsonl(self._validation_dir / "rejections.jsonl")[-limit:]

    def get_recent_settlement_audit(self, limit: int = 50) -> List[dict]:
        if self._validation_dir is None:
            return []
        return self._read_jsonl(self._validation_dir / "settlement_audit.jsonl")[-limit:]

    def get_validation_context(self) -> dict:
        return {
            "validation_run_id": self._validation_run_id,
            "fresh_book_started_at": self._fresh_book_started_at,
            "archived_state_path": self._archived_state_path,
            "validation_mode": bool(config.FUNDING_VALIDATION_MODE),
            "validation_scope": str(getattr(config, "FUNDING_VALIDATION_SCOPE", "hedge_only")),
        }

    def add_position(self, position: HedgePosition) -> None:
        """Add a new open position."""
        if self._validation_run_id and not position.validation_run_id:
            position.validation_run_id = self._validation_run_id
        self._positions[position.symbol] = position
        self._save_state()
        logger.info("Opened position: %s", position.symbol)

    def open_positions(self) -> List[HedgePosition]:
        """Return all currently open positions."""
        return [
            p for p in self._positions.values()
            if p.status == HedgeStatus.OPEN
        ]

    def get_position(self, symbol: str) -> Optional[HedgePosition]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def total_exposure(self) -> Decimal:
        """Sum of notional values of all open positions."""
        return sum(
            p.notional_value()
            for p in self._positions.values()
            if p.status == HedgeStatus.OPEN
        )

    def record_funding(
        self,
        symbol: str,
        amount: Decimal,
        timestamp: Optional[datetime] = None,
        expected_payment: Optional[Decimal] = None,
        cap_applied: bool = False,
    ) -> None:
        """Record a funding payment for a position."""
        pos = self._positions.get(symbol)
        if pos and pos.status == HedgeStatus.OPEN:
            pos.funding_collected += amount
            pos.realized_funding_events += 1
            if expected_payment is not None:
                pos.expected_funding_payment = expected_payment
            if cap_applied:
                pos.funding_cap_applied = True
            self._save_state()
            logger.info(
                "Recorded funding $%s for %s (total: $%s)",
                amount, symbol, pos.funding_collected,
            )

    def update_position(self, symbol: str, **fields) -> Optional[HedgePosition]:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        for key, value in fields.items():
            if hasattr(pos, key):
                setattr(pos, key, value)
        self._save_state()
        return pos

    def close_position(
        self,
        symbol: str,
        exit_price_spot: Decimal = Decimal("0"),
        exit_price_perp: Decimal = Decimal("0"),
        exit_pnl: Decimal = Decimal("0"),
        trading_fees: Decimal = Decimal("0"),
    ) -> Optional[HedgePosition]:
        """Close a position with exit data."""
        pos = self._positions.get(symbol)
        if pos is None:
            logger.warning("No position found for %s", symbol)
            return None

        pos.status = HedgeStatus.CLOSED
        pos.exit_time = datetime.now(timezone.utc)
        pos.exit_price_spot = exit_price_spot
        pos.exit_price_perp = exit_price_perp
        pos.exit_pnl = exit_pnl
        pos.trading_fees_paid += trading_fees
        self._save_state()

        logger.info(
            "Closed %s: exit PnL=$%s, funding=$%s, fees=$%s, net=$%s",
            symbol, exit_pnl, pos.funding_collected,
            pos.trading_fees_paid, pos.net_pnl(),
        )
        return pos

    def all_positions(self) -> List[HedgePosition]:
        """Return all positions (open and closed)."""
        return list(self._positions.values())
