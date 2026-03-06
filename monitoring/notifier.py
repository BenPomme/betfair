from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import config
from monitoring.transports.discord_webhook import discord_configured, send_discord

logger = logging.getLogger(__name__)


@dataclass
class NotificationState:
    last_notification_ts: Optional[str] = None
    last_digest_ts: Optional[str] = None
    last_daily_digest_ts: Optional[str] = None
    notification_failures: int = 0
    discord_configured: bool = False
    sent_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_notification_ts": self.last_notification_ts,
            "last_digest_ts": self.last_digest_ts,
            "last_daily_digest_ts": self.last_daily_digest_ts,
            "notification_failures": self.notification_failures,
            "discord_configured": self.discord_configured,
            "sent_events": self.sent_events[-50:],
        }


class NotificationManager:
    def __init__(self) -> None:
        self._dedupe: Dict[str, float] = {}
        self._state = NotificationState(discord_configured=discord_configured())

    def _utc_now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def state(self) -> Dict[str, Any]:
        self._state.discord_configured = discord_configured()
        return self._state.to_dict()

    def _should_send(self, dedupe_key: str) -> bool:
        now = time.time()
        last_ts = self._dedupe.get(dedupe_key)
        if last_ts is None:
            self._dedupe[dedupe_key] = now
            return True
        if (now - last_ts) >= max(60, int(config.NOTIFY_DEDUPE_WINDOW_SECONDS)):
            self._dedupe[dedupe_key] = now
            return True
        return False

    def send_event(
        self,
        *,
        portfolio_id: str,
        severity: str,
        event_type: str,
        title: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        dedupe_key: Optional[str] = None,
        allow_unlisted: bool = False,
    ) -> bool:
        if not config.NOTIFICATIONS_ENABLED:
            return False
        if not discord_configured():
            self._state.discord_configured = False
            return False
        if config.DISCORD_NOTIFY_CRITICAL_ONLY and severity != "critical":
            return False
        allowed = {p.strip() for p in str(config.DISCORD_NOTIFY_PORTFOLIOS).split(",") if p.strip()}
        if allowed and portfolio_id not in allowed and not allow_unlisted:
            return False
        key = dedupe_key or hashlib.sha1(
            f"{portfolio_id}|{severity}|{event_type}|{title}|{message}".encode("utf-8")
        ).hexdigest()
        if not self._should_send(key):
            return False
        body = f"[{severity.upper()}] {title}\nportfolio={portfolio_id}\n{message}"
        ok = send_discord(body)
        if ok:
            self._state.last_notification_ts = self._utc_now_iso()
            self._state.sent_events.append(
                {
                    "ts": self._state.last_notification_ts,
                    "portfolio_id": portfolio_id,
                    "severity": severity,
                    "event_type": event_type,
                    "title": title,
                    "message": message,
                    "payload": payload or {},
                }
            )
        else:
            self._state.notification_failures += 1
        return ok

    def send_digest(self, summaries: Iterable[Dict[str, Any]]) -> bool:
        if not (config.NOTIFICATIONS_ENABLED and config.DISCORD_DIGEST_ENABLED):
            return False
        if not discord_configured():
            self._state.discord_configured = False
            return False
        lines = ["Portfolio digest"]
        for item in summaries:
            lines.append(
                f"- {item.get('label', item.get('portfolio_id'))}: "
                f"status={item.get('status')} readiness={item.get('readiness')} "
                f"roi={item.get('roi_pct', 0):.2f}% "
                f"pnl={item.get('realized_pnl', 0):.2f} {item.get('currency', '')} "
                f"open={item.get('open_count', 0)}"
            )
        ok = send_discord("\n".join(lines), username="Strategy Digest")
        if ok:
            self._state.last_digest_ts = self._utc_now_iso()
        else:
            self._state.notification_failures += 1
        return ok

    def send_daily_digest(self, lines: List[str]) -> bool:
        if not (config.NOTIFICATIONS_ENABLED and config.DISCORD_DAILY_DIGEST_ENABLED):
            return False
        if not discord_configured():
            self._state.discord_configured = False
            return False
        ok = send_discord("\n".join(lines), username="Strategy Daily Digest")
        if ok:
            self._state.last_daily_digest_ts = self._utc_now_iso()
        else:
            self._state.notification_failures += 1
        return ok
