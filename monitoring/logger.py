"""
Structured JSON logging for trade log, opportunities, errors.
Output to file or PostgreSQL via config DATABASE_URL.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)


def log_trade(entry: dict) -> None:
    """Write a trade/opportunity log entry as JSON."""
    line = json.dumps(entry) + "\n"
    log_path = os.getenv("TRADE_LOG_PATH", "logs/trades.jsonl")
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(line)
    except Exception as e:
        logger.exception("Failed to write trade log: %s", e)


def log_opportunity(market_id: str, net_profit_eur: float, overround: float) -> None:
    """Convenience: log a detected opportunity (before execution)."""
    log_trade({
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "opportunity",
        "market_id": market_id,
        "net_profit_eur": net_profit_eur,
        "overround": overround,
    })


def log_error(context: str, error: str) -> None:
    """Log an error with context."""
    log_trade({
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "error",
        "context": context,
        "error": error,
    })
