from __future__ import annotations

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def discord_configured() -> bool:
    return bool(config.DISCORD_ENABLED and str(config.DISCORD_WEBHOOK_URL).strip())


def send_discord(message: str, *, username: Optional[str] = None) -> bool:
    if not discord_configured():
        logger.debug("Discord not configured; skipping message")
        return False
    try:
        import httpx

        payload = {"content": message[:1900]}
        if username:
            payload["username"] = username
        elif config.DISCORD_WEBHOOK_USERNAME:
            payload["username"] = config.DISCORD_WEBHOOK_USERNAME
        if config.DISCORD_WEBHOOK_AVATAR_URL:
            payload["avatar_url"] = config.DISCORD_WEBHOOK_AVATAR_URL
        response = httpx.post(
            config.DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10.0,
        )
        return 200 <= response.status_code < 300
    except Exception as exc:
        logger.exception("Discord send failed: %s", exc)
        return False
