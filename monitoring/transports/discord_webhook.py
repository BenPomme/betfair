from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def discord_configured() -> bool:
    return bool(config.DISCORD_ENABLED and str(config.DISCORD_WEBHOOK_URL).strip())


def send_discord(
    message: str = "",
    *,
    username: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    if not discord_configured():
        logger.debug("Discord not configured; skipping message")
        return False
    try:
        import httpx

        payload: Dict[str, Any] = {}
        if message:
            payload["content"] = message[:1900]
        if embeds:
            payload["embeds"] = embeds[:10]
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
