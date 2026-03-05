"""
Directional position manager: track open contrarian positions with JSON persistence.
In-memory list + file-based state. Mirrors PositionManager pattern for HedgePosition
but adapted to DirectionalPosition (contrarian / unhedged directional trades).

State is persisted to config.CONTRARIAN_STATE_PATH (default: data/state/directional_positions.json).
"""
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

import config
from funding.core.schemas import DirectionalPosition, DirectionalPositionStatus

logger = logging.getLogger(__name__)


class DirectionalPositionManager:
    """Track open and closed directional (contrarian) positions."""

    def __init__(self, state_path: Optional[str] = None) -> None:
        self._state_path = Path(state_path or config.CONTRARIAN_STATE_PATH)
        self._positions: List[DirectionalPosition] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load positions from JSON file on init. Missing file is treated as empty state."""
        if not self._state_path.exists():
            logger.debug(
                "No directional position state file at %s; starting with empty state",
                self._state_path,
            )
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._positions = [
                DirectionalPosition.from_dict(pos_data)
                for pos_data in raw.get("positions", [])
            ]
            logger.info(
                "Loaded %d directional positions from state (%s open)",
                len(self._positions),
                sum(1 for p in self._positions if p.status == DirectionalPositionStatus.OPEN),
            )
        except Exception as exc:
            logger.warning(
                "Failed to load directional position state from %s: %s",
                self._state_path,
                exc,
            )

    def _save(self) -> None:
        """Persist all positions to JSON."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "positions": [p.to_dict() for p in self._positions],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_position(self, position: DirectionalPosition) -> None:
        """Append a new open position and persist state."""
        self._positions.append(position)
        self._save()
        logger.info(
            "Opened directional position: %s %s @ %s (qty=%s, stop=%s, tp=%s)",
            position.side.value,
            position.symbol,
            position.entry_price,
            position.quantity,
            position.stop_loss,
            position.take_profit,
        )

    def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        realized_pnl: Decimal,
        trading_fees: Decimal,
        status: DirectionalPositionStatus,
    ) -> Optional[DirectionalPosition]:
        """
        Close the open position for *symbol*.

        Updates exit_price, exit_time, realized_pnl, trading_fees_paid, and status,
        then persists state.

        Args:
            symbol:       Symbol to close (must match an OPEN position).
            exit_price:   Price at which the position was closed.
            realized_pnl: Gross P&L from the price move (before fees).
            trading_fees: Total trading fees paid on entry + exit.
            status:       Closing status (CLOSED, STOPPED, TOOK_PROFIT, etc.).

        Returns:
            The updated DirectionalPosition, or None if no open position found.
        """
        pos = self.get_position(symbol)
        if pos is None:
            logger.warning(
                "close_position: no open position found for %s", symbol
            )
            return None

        pos.exit_price = exit_price
        pos.exit_time = datetime.now(timezone.utc)
        pos.realized_pnl = realized_pnl
        pos.trading_fees_paid += trading_fees
        pos.status = status
        self._save()

        net = realized_pnl - trading_fees
        logger.info(
            "Closed %s %s: status=%s, exit_price=%s, realized_pnl=%s, fees=%s, net=%s",
            pos.side.value,
            symbol,
            status.value,
            exit_price,
            realized_pnl,
            trading_fees,
            net,
        )
        return pos

    def update_position(self, position: DirectionalPosition) -> None:
        """Persist in-place changes to an existing position object."""
        self._save()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def open_positions(self) -> List[DirectionalPosition]:
        """Return all positions with status == OPEN."""
        return [
            p for p in self._positions
            if p.status == DirectionalPositionStatus.OPEN
        ]

    def get_position(self, symbol: str) -> Optional[DirectionalPosition]:
        """
        Find the open position for *symbol*.

        Returns:
            The first DirectionalPosition with status OPEN and matching symbol,
            or None if not found.
        """
        for pos in self._positions:
            if pos.symbol == symbol and pos.status == DirectionalPositionStatus.OPEN:
                return pos
        return None

    def total_exposure(self) -> Decimal:
        """Sum of notional_value() for all open positions."""
        return sum(
            p.notional_value()
            for p in self._positions
            if p.status == DirectionalPositionStatus.OPEN
        )

    def all_positions(self) -> List[DirectionalPosition]:
        """Return all positions (open and closed)."""
        return list(self._positions)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def win_rate(self) -> float:
        """
        Win rate of closed positions.

        A closed position is a "win" if realized_pnl > 0 (gross, before fees).
        Returns 0.0 if there are no closed positions.
        """
        closed = [
            p for p in self._positions
            if p.status != DirectionalPositionStatus.OPEN
            and p.status != DirectionalPositionStatus.CLOSING
        ]
        if not closed:
            return 0.0
        wins = sum(1 for p in closed if p.realized_pnl > Decimal("0"))
        return wins / len(closed)

    def avg_hold_hours(self) -> float:
        """
        Average hold duration in hours for closed positions.

        Only positions with both entry_time and exit_time are included.
        Returns 0.0 if no closed positions have valid timestamps.
        """
        durations = []
        for pos in self._positions:
            if (
                pos.status not in (DirectionalPositionStatus.OPEN, DirectionalPositionStatus.CLOSING)
                and pos.entry_time is not None
                and pos.exit_time is not None
            ):
                delta = pos.exit_time - pos.entry_time
                durations.append(delta.total_seconds() / 3600.0)
        if not durations:
            return 0.0
        return sum(durations) / len(durations)
