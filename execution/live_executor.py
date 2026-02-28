"""
Live executor: place limit orders via Betfair API. Used only when PAPER_TRADING=false.
Retry logic, partial fill handling; on failure of one leg attempt cancel of others.
"""
import logging
from decimal import Decimal
from typing import Optional, List, Any

from core.types import Opportunity

logger = logging.getLogger(__name__)

try:
    import betfairlightweight
    HAS_BETFAIR = True
except ImportError:
    HAS_BETFAIR = False


class LiveExecutor:
    """Place real orders on Betfair. Only use after paper gate passed."""

    def __init__(self, client: Any = None):
        self._client = client
        self._max_retries = 3

    def place(self, opportunity: Opportunity) -> Optional[dict]:
        """
        Place limit back orders for all selections. On partial failure cancel other legs.
        Returns dict with order_ids or None on failure.
        """
        if not HAS_BETFAIR:
            raise NotImplementedError("betfairlightweight not installed")
        if self._client is None:
            raise NotImplementedError(
                "Live execution requires Betfair API client. "
                "Keep PAPER_TRADING=true until paper gate passed."
            )

        market_id = opportunity.market_id
        order_ids: List[str] = []
        placed_indices: List[int] = []

        try:
            for i, sel in enumerate(opportunity.selections):
                selection_id = self._selection_id_from_opportunity(opportunity, i)
                price = Decimal(str(sel["back_price"]))
                size = Decimal(str(sel["stake_eur"]))
                for attempt in range(self._max_retries):
                    try:
                        order = self._place_back_order(
                            market_id=market_id,
                            selection_id=selection_id,
                            price=price,
                            size=size,
                        )
                        if order:
                            order_ids.append(order)
                            placed_indices.append(i)
                        break
                    except Exception as e:
                        logger.warning("Place attempt %s failed: %s", attempt + 1, e)
                        if attempt == self._max_retries - 1:
                            raise
            return {"order_ids": order_ids, "market_id": market_id}
        except Exception as e:
            logger.exception("Live execution failed: %s", e)
            if order_ids:
                self._cancel_orders(order_ids)
            raise

    def _selection_id_from_opportunity(self, opportunity: Opportunity, index: int) -> str:
        """Resolve selection_id from opportunity (Betfair runner id)."""
        sel = opportunity.selections[index]
        return str(sel.get("selection_id", index + 1))

    def _place_back_order(
        self,
        market_id: str,
        selection_id: str,
        price: Decimal,
        size: Decimal,
    ) -> Optional[str]:
        """Place a single back limit order. Returns bet_id or None."""
        limit_order = betfairlightweight.filters.limit_order(
            size=float(size),
            price=float(price),
            persistence_type="LAPSE",
        )
        instruction = betfairlightweight.filters.place_instruction(
            order_type="LIMIT",
            selection_id=int(selection_id),
            side="BACK",
            limit_order=limit_order,
        )
        result = self._client.betting.place_orders(
            market_id=market_id,
            instructions=[instruction],
        )
        if result and getattr(result, "status", None) == "SUCCESS":
            reports = getattr(result, "place_instruction_reports", []) or []
            if reports and getattr(reports[0], "bet_id", None):
                return reports[0].bet_id
        return None

    def _cancel_orders(self, order_ids: List[str]) -> None:
        """Attempt to cancel given orders."""
        try:
            instructions = [
                betfairlightweight.filters.cancel_instruction(bet_id=bid)
                for bid in order_ids
            ]
            self._client.betting.cancel_orders(
                market_id=None,
                instructions=instructions,
            )
        except Exception as e:
            logger.exception("Cancel orders failed: %s", e)
