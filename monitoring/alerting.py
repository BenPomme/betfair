"""
Telegram Bot API alerts: execution failure, circuit breaker, daily loss cap.
"""
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    """Send message to TELEGRAM_CHAT_ID via TELEGRAM_BOT_TOKEN. Returns True if sent."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured; skipping alert")
        return False
    try:
        import httpx
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = httpx.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10.0)
        return resp.status_code == 200
    except Exception as e:
        logger.exception("Telegram send failed: %s", e)
        return False


def alert_execution_failure(market_id: str, error: str) -> None:
    send_telegram(f"[Arb] Execution failed market_id={market_id}: {error}")


def alert_circuit_breaker() -> None:
    send_telegram("[Arb] Circuit breaker: trading halted after 3 consecutive failures.")


def alert_daily_loss_cap() -> None:
    send_telegram("[Arb] Daily loss limit reached. Trading halted.")
