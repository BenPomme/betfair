"""
Shared types for the funding rate arbitrage pipeline.
All monetary values use Decimal. Timestamps use datetime.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional


class HedgeStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FundingSnapshot:
    """Point-in-time funding and price data for a perpetual symbol."""
    symbol: str
    funding_rate: Decimal          # current funding rate (e.g. 0.0001 = 0.01%)
    next_funding_time: datetime    # next settlement timestamp
    mark_price: Decimal
    index_price: Decimal
    open_interest: Decimal         # in contracts/coins
    timestamp: datetime            # when this snapshot was taken


@dataclass(frozen=True)
class FundingOpportunity:
    """A detected funding arbitrage opportunity ready for evaluation."""
    symbol: str
    current_rate: Decimal          # current funding rate per 8h
    predicted_rate: Decimal        # ML-predicted next rate (Phase 1: same as current)
    annualized_yield: Decimal      # rate × 3 × 365
    entry_price_spot: Decimal      # current spot price
    entry_price_perp: Decimal      # current perp mark price
    position_size: Decimal         # notional USD
    expected_funding_payment: Decimal  # single settlement payment
    timestamp: datetime


@dataclass
class HedgePosition:
    """A live or historical delta-neutral hedge position."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side_spot: str = "BUY"         # always BUY for cash-and-carry
    side_perp: str = "SELL"        # always SELL (short) for cash-and-carry
    entry_price_spot: Decimal = Decimal("0")
    entry_price_perp: Decimal = Decimal("0")
    quantity_spot: Decimal = Decimal("0")
    quantity_perp: Decimal = Decimal("0")
    leverage: int = 2
    margin_type: str = "ISOLATED"
    entry_time: Optional[datetime] = None
    funding_collected: Decimal = Decimal("0")
    trading_fees_paid: Decimal = Decimal("0")
    status: HedgeStatus = HedgeStatus.OPEN
    exit_time: Optional[datetime] = None
    exit_price_spot: Decimal = Decimal("0")
    exit_price_perp: Decimal = Decimal("0")
    exit_pnl: Decimal = Decimal("0")

    def notional_value(self) -> Decimal:
        """Current notional value of the position (spot side)."""
        return self.entry_price_spot * self.quantity_spot

    def net_pnl(self) -> Decimal:
        """Total P&L including funding collected minus fees."""
        return self.funding_collected + self.exit_pnl - self.trading_fees_paid

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side_spot": self.side_spot,
            "side_perp": self.side_perp,
            "entry_price_spot": str(self.entry_price_spot),
            "entry_price_perp": str(self.entry_price_perp),
            "quantity_spot": str(self.quantity_spot),
            "quantity_perp": str(self.quantity_perp),
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "funding_collected": str(self.funding_collected),
            "trading_fees_paid": str(self.trading_fees_paid),
            "status": self.status.value,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price_spot": str(self.exit_price_spot),
            "exit_price_perp": str(self.exit_price_perp),
            "exit_pnl": str(self.exit_pnl),
        }

    @classmethod
    def from_dict(cls, d: dict) -> HedgePosition:
        """Deserialize from JSON."""
        return cls(
            id=d["id"],
            symbol=d["symbol"],
            side_spot=d.get("side_spot", "BUY"),
            side_perp=d.get("side_perp", "SELL"),
            entry_price_spot=Decimal(d["entry_price_spot"]),
            entry_price_perp=Decimal(d["entry_price_perp"]),
            quantity_spot=Decimal(d["quantity_spot"]),
            quantity_perp=Decimal(d["quantity_perp"]),
            leverage=int(d.get("leverage", 2)),
            margin_type=d.get("margin_type", "ISOLATED"),
            entry_time=datetime.fromisoformat(d["entry_time"]) if d.get("entry_time") else None,
            funding_collected=Decimal(d.get("funding_collected", "0")),
            trading_fees_paid=Decimal(d.get("trading_fees_paid", "0")),
            status=HedgeStatus(d.get("status", "OPEN")),
            exit_time=datetime.fromisoformat(d["exit_time"]) if d.get("exit_time") else None,
            exit_price_spot=Decimal(d.get("exit_price_spot", "0")),
            exit_price_perp=Decimal(d.get("exit_price_perp", "0")),
            exit_pnl=Decimal(d.get("exit_pnl", "0")),
        )


# ---------------------------------------------------------------------------
# Directional (contrarian) position types
# ---------------------------------------------------------------------------


class DirectionalSide(enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class DirectionalPositionStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"
    TOOK_PROFIT = "TOOK_PROFIT"


@dataclass(frozen=True)
class ContrarianSignal:
    """Immutable signal produced by the contrarian model for a single symbol."""
    symbol: str
    direction: DirectionalSide
    confidence: float
    predicted_return_24h: float
    predicted_return_72h: float
    model_name: str
    funding_rate: Decimal
    long_short_ratio: Optional[float] = None
    fear_greed: Optional[int] = None
    regime: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DirectionalPosition:
    """A live or historical directional (non-delta-neutral) position."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: DirectionalSide = DirectionalSide.LONG
    entry_price: Decimal = Decimal("0")
    exit_price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    leverage: int = 2
    margin_type: str = "ISOLATED"
    stop_loss: Decimal = Decimal("0")
    take_profit: Decimal = Decimal("0")
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    status: DirectionalPositionStatus = DirectionalPositionStatus.OPEN
    realized_pnl: Decimal = Decimal("0")
    trading_fees_paid: Decimal = Decimal("0")
    signal: Optional[ContrarianSignal] = None

    def notional_value(self) -> Decimal:
        """Notional value of the position at entry."""
        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Unrealized P&L at the given current price."""
        if self.side is DirectionalSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        signal_dict: Optional[dict] = None
        if self.signal is not None:
            signal_dict = {
                "symbol": self.signal.symbol,
                "direction": self.signal.direction.value,
                "confidence": self.signal.confidence,
                "predicted_return_24h": self.signal.predicted_return_24h,
                "predicted_return_72h": self.signal.predicted_return_72h,
                "model_name": self.signal.model_name,
                "funding_rate": str(self.signal.funding_rate),
                "long_short_ratio": self.signal.long_short_ratio,
                "fear_greed": self.signal.fear_greed,
                "regime": self.signal.regime,
                "timestamp": self.signal.timestamp.isoformat(),
            }
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "quantity": str(self.quantity),
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "stop_loss": str(self.stop_loss),
            "take_profit": str(self.take_profit),
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status.value,
            "realized_pnl": str(self.realized_pnl),
            "trading_fees_paid": str(self.trading_fees_paid),
            "signal": signal_dict,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DirectionalPosition:
        """Deserialize from JSON."""
        signal: Optional[ContrarianSignal] = None
        if d.get("signal"):
            s = d["signal"]
            signal = ContrarianSignal(
                symbol=s["symbol"],
                direction=DirectionalSide(s["direction"]),
                confidence=float(s["confidence"]),
                predicted_return_24h=float(s["predicted_return_24h"]),
                predicted_return_72h=float(s["predicted_return_72h"]),
                model_name=s["model_name"],
                funding_rate=Decimal(s["funding_rate"]),
                long_short_ratio=s.get("long_short_ratio"),
                fear_greed=s.get("fear_greed"),
                regime=s.get("regime"),
                timestamp=datetime.fromisoformat(s["timestamp"]),
            )
        return cls(
            id=d["id"],
            symbol=d["symbol"],
            side=DirectionalSide(d.get("side", "LONG")),
            entry_price=Decimal(d.get("entry_price", "0")),
            exit_price=Decimal(d.get("exit_price", "0")),
            quantity=Decimal(d.get("quantity", "0")),
            leverage=int(d.get("leverage", 2)),
            margin_type=d.get("margin_type", "ISOLATED"),
            stop_loss=Decimal(d.get("stop_loss", "0")),
            take_profit=Decimal(d.get("take_profit", "0")),
            entry_time=datetime.fromisoformat(d["entry_time"]) if d.get("entry_time") else None,
            exit_time=datetime.fromisoformat(d["exit_time"]) if d.get("exit_time") else None,
            status=DirectionalPositionStatus(d.get("status", "OPEN")),
            realized_pnl=Decimal(d.get("realized_pnl", "0")),
            trading_fees_paid=Decimal(d.get("trading_fees_paid", "0")),
            signal=signal,
        )
