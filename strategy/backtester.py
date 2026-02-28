"""
Replay historical stream files; run scanner + paper executor to validate thresholds.
Optional for MVP; Phase 4.
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Callable, Any
import json

from core.types import PriceSnapshot, SelectionPrice
from core.scanner import scan_snapshot
from data.price_cache import PriceCache
from execution.paper_executor import PaperExecutor


def load_snapshot_from_dict(data: dict) -> PriceSnapshot:
    """Build PriceSnapshot from a dict (e.g. one line of a stream replay file)."""
    market_id = data.get("market_id", "")
    ts = data.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts is None:
        ts = datetime.now(timezone.utc)
    selections = []
    for s in data.get("selections", []):
        selections.append(SelectionPrice(
            selection_id=str(s.get("selection_id", "")),
            name=str(s.get("name", "")),
            best_back_price=Decimal(str(s.get("best_back_price", 0))),
            available_to_back=Decimal(str(s.get("available_to_back", 0))),
        ))
    return PriceSnapshot(market_id=market_id, selections=tuple(selections), timestamp=ts)


def run_backtest(
    replay_path: str,
    paper_executor: Optional[PaperExecutor] = None,
) -> dict:
    """
    Read replay file (one JSON object per line, each with market_id, timestamp, selections),
    run scan_snapshot on each; on opportunity run paper_executor.log.
    Returns summary: num_snapshots, num_opportunities, total_net_profit.
    """
    if paper_executor is None:
        paper_executor = PaperExecutor()
    num_snapshots = 0
    num_opportunities = 0
    total_net_profit = Decimal("0")
    try:
        with open(replay_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                snap = load_snapshot_from_dict(data)
                num_snapshots += 1
                opp = scan_snapshot(snap)
                if opp:
                    num_opportunities += 1
                    paper_executor.log(opp)
                    total_net_profit += opp.net_profit_eur
    except FileNotFoundError:
        return {
            "num_snapshots": 0,
            "num_opportunities": 0,
            "total_net_profit": 0,
            "error": "Replay file not found",
        }
    return {
        "num_snapshots": num_snapshots,
        "num_opportunities": num_opportunities,
        "total_net_profit": float(total_net_profit),
        "log_entries": len(paper_executor.log_entries),
    }
