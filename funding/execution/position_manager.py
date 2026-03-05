"""
Position manager: track open hedge positions with JSON persistence.
In-memory cache + file-based state (no DB dependency for Phase 1).
"""
import json
import logging
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
        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
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
        }
        self._state_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def add_position(self, position: HedgePosition) -> None:
        """Add a new open position."""
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
        self, symbol: str, amount: Decimal, timestamp: Optional[datetime] = None
    ) -> None:
        """Record a funding payment for a position."""
        pos = self._positions.get(symbol)
        if pos and pos.status == HedgeStatus.OPEN:
            pos.funding_collected += amount
            self._save_state()
            logger.info(
                "Recorded funding $%s for %s (total: $%s)",
                amount, symbol, pos.funding_collected,
            )

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
