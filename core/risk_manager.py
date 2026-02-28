"""
Exposure limits, daily loss cap, max open bets. Gate before execution.
"""
from decimal import Decimal
from typing import Any, Optional

from core.types import Opportunity


class RiskManager:
    """Checks whether an opportunity is within risk limits."""

    def __init__(
        self,
        max_stake_eur: Decimal,
        daily_loss_limit_eur: Decimal,
        max_open_bets: int = 10,
    ):
        self.max_stake_eur = max_stake_eur
        self.daily_loss_limit_eur = daily_loss_limit_eur
        self.max_open_bets = max_open_bets
        self._open_bets = 0
        self._daily_pnl_eur = Decimal("0")

    def can_execute(
        self,
        opportunity: Opportunity,
        context: Optional[dict] = None,
    ) -> bool:
        """
        Return True if the opportunity is within limits:
        open bets < max_open_bets, daily P&L above -daily_loss_limit.
        Note: stake sizing is handled by proportional staking in the scan loop,
        so we don't enforce max_stake_eur here (it would conflict with dynamic sizing).
        """
        if self._open_bets >= self.max_open_bets:
            return False
        if self._daily_pnl_eur <= -self.daily_loss_limit_eur:
            return False
        return True

    def register_execution(self, opportunity: Opportunity, net_pnl_eur: Decimal) -> None:
        """Call after executing (paper or live) to update open bets and daily P&L."""
        self._open_bets += 1
        self._daily_pnl_eur += net_pnl_eur

    def register_settlement(self, net_pnl_eur: Decimal) -> None:
        """Call when a bet settles; decrement open bets, update daily P&L."""
        self._open_bets = max(0, self._open_bets - 1)
        self._daily_pnl_eur += net_pnl_eur

    def set_open_bets(self, n: int) -> None:
        """For testing or reset."""
        self._open_bets = n

    def set_daily_pnl(self, pnl: Decimal) -> None:
        """For testing or daily reset."""
        self._daily_pnl_eur = pnl
