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

    @staticmethod
    def _severity_color(severity: str) -> int:
        return {
            "critical": 0xD33F49,
            "warning": 0xE0A100,
            "info": 0x2F80ED,
        }.get(str(severity or "").lower(), 0x6C7A89)

    @staticmethod
    def _status_badge(value: str) -> str:
        mapping = {
            "live_ready": "LIVE READY",
            "candidate": "CANDIDATE",
            "paper_validating": "PAPER VALIDATING",
            "research_only": "RESEARCH ONLY",
            "blocked": "NOT LIVE READY",
            "ready": "READY",
            "validated": "VALIDATED",
        }
        key = str(value or "").strip().lower()
        return mapping.get(key, str(value or "unknown").replace("_", " ").upper())

    @staticmethod
    def _signed_number(value: float, suffix: str = "") -> str:
        return f"{value:+.2f}{suffix}"

    def _trade_threshold(self, currency: str) -> float:
        code = str(currency or "").upper()
        if code == "EUR":
            return float(config.DISCORD_MIN_TRADE_ALERT_PNL_EUR)
        return float(config.DISCORD_MIN_TRADE_ALERT_PNL_USD)

    def _event_embed(
        self,
        *,
        portfolio_id: str,
        severity: str,
        event_type: str,
        title: str,
        message: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        embed: Dict[str, Any] = {
            "title": title,
            "description": message,
            "color": self._severity_color(severity),
            "fields": [
                {"name": "Portfolio", "value": str(payload.get("portfolio_label") or portfolio_id), "inline": True},
                {"name": "Type", "value": str(event_type).replace("_", " ").upper(), "inline": True},
                {"name": "At", "value": self._utc_now_iso(), "inline": True},
            ],
        }
        if event_type == "trade_closed":
            pnl = float(payload.get("pnl", 0.0) or 0.0)
            currency = str(payload.get("currency", ""))
            embed["fields"].extend(
                [
                    {"name": "Outcome", "value": "WIN" if pnl >= 0 else "LOSS", "inline": True},
                    {"name": "P&L", "value": f"{pnl:+.2f} {currency}".strip(), "inline": True},
                    {"name": "Reason", "value": str(payload.get("close_reason") or "closed"), "inline": True},
                    {"name": "Book", "value": f"ROI {float(payload.get('roi_pct', 0.0) or 0.0):+.2f}% | Realized {float(payload.get('book_realized_pnl', 0.0) or 0.0):+.2f} {currency}".strip(), "inline": False},
                    {"name": "Trust", "value": f"{self._status_badge(str(payload.get('readiness') or 'unknown'))} | Progress {float(payload.get('progress_pct', 0.0) or 0.0):.0f}%", "inline": False},
                ]
            )
        elif event_type == "model_update":
            embed["fields"].extend(
                [
                    {"name": "Model", "value": str(payload.get("model_id") or "model"), "inline": True},
                    {"name": "Gate", "value": "PASS" if payload.get("strict_gate_pass") else "FAIL", "inline": True},
                    {"name": "Result", "value": str(payload.get("last_retrain_result") or "updated"), "inline": True},
                    {"name": "AUC", "value": str(payload.get("current_auc")), "inline": True},
                    {"name": "Brier Lift", "value": str(payload.get("rolling_200_brier_lift")), "inline": True},
                    {"name": "Settled", "value": str(payload.get("settled_count")), "inline": True},
                ]
            )
        elif event_type == "readiness_changed":
            blockers = payload.get("blockers") or []
            if blockers:
                embed["fields"].append({"name": "Blockers", "value": ", ".join(map(str, blockers[:5])), "inline": False})
        elif event_type == "deploy_updated":
            embed["fields"].extend(
                [
                    {"name": "Commit", "value": str(payload.get("new_head") or "unknown"), "inline": True},
                    {"name": "Branch", "value": str(payload.get("branch") or "main"), "inline": True},
                    {"name": "Portfolios", "value": ", ".join(payload.get("restarted_portfolios") or []) or "none", "inline": False},
                ]
            )
        return embed

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
        event_payload = dict(payload or {})
        body = f"[{severity.upper()}] {title}\nportfolio={portfolio_id}\n{message}"
        embed = self._event_embed(
            portfolio_id=portfolio_id,
            severity=severity,
            event_type=event_type,
            title=title,
            message=message,
            payload=event_payload,
        )
        ok = send_discord(body, embeds=[embed])
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

    def send_digest(self, summaries: Iterable[Dict[str, Any]], *, snapshots: Optional[Iterable[Dict[str, Any]]] = None) -> bool:
        if not (config.NOTIFICATIONS_ENABLED and config.DISCORD_DIGEST_ENABLED):
            return False
        if not discord_configured():
            self._state.discord_configured = False
            return False
        summary_list = list(summaries)
        snapshot_map = {
            item["summary"]["portfolio_id"]: item
            for item in (snapshots or [])
            if isinstance(item, dict) and item.get("summary")
        }
        leaders = sorted(summary_list, key=lambda item: float(item.get("realized_pnl", 0.0) or 0.0), reverse=True)
        improvers = sorted(summary_list, key=lambda item: float(item.get("progress_delta_24h", 0.0) or 0.0), reverse=True)
        blockers = [item for item in summary_list if int(item.get("blocker_count", 0) or 0) > 0]
        lines = ["Portfolio pulse"]
        for item in leaders[:5]:
            lines.append(
                f"- {item.get('label')}: {self._status_badge(item.get('readiness'))} | "
                f"PnL {float(item.get('realized_pnl', 0.0) or 0.0):+.2f} {item.get('currency', '')} | "
                f"ROI {float(item.get('roi_pct', 0.0) or 0.0):+.2f}%"
            )
        embed = {
            "title": "Portfolio Pulse",
            "description": "Decision-grade summary",
            "color": self._severity_color("info"),
            "fields": [
                {
                    "name": "Leaders",
                    "value": "\n".join(
                        f"{item.get('label')}: {float(item.get('realized_pnl', 0.0) or 0.0):+.2f} {item.get('currency', '')} | {float(item.get('roi_pct', 0.0) or 0.0):+.2f}%"
                        for item in leaders[:3]
                    ) or "none",
                    "inline": False,
                },
                {
                    "name": "Improving",
                    "value": "\n".join(
                        f"{item.get('label')}: {float(item.get('progress_delta_24h', 0.0) or 0.0):+.1f} pts | {self._status_badge(item.get('readiness'))}"
                        for item in improvers[:3]
                    ) or "none",
                    "inline": False,
                },
                {
                    "name": "Main Blockers",
                    "value": "\n".join(
                        f"{item.get('label')}: {int(item.get('blocker_count', 0) or 0)} blockers"
                        for item in blockers[:3]
                    ) or "none",
                    "inline": False,
                },
            ],
        }
        ok = send_discord("\n".join(lines), username="Strategy Digest", embeds=[embed])
        if ok:
            self._state.last_digest_ts = self._utc_now_iso()
        else:
            self._state.notification_failures += 1
        return ok

    def send_daily_digest(self, lines: List[str], *, sections: Optional[Dict[str, List[str]]] = None) -> bool:
        if not (config.NOTIFICATIONS_ENABLED and config.DISCORD_DAILY_DIGEST_ENABLED):
            return False
        if not discord_configured():
            self._state.discord_configured = False
            return False
        embed_fields = []
        for title, items in (sections or {}).items():
            if not items:
                continue
            embed_fields.append({"name": title, "value": "\n".join(items[:5]), "inline": False})
        embed = {
            "title": "Daily Strategy Summary",
            "description": lines[0] if lines else "Daily summary",
            "color": self._severity_color("info"),
            "fields": embed_fields[:5],
        }
        ok = send_discord("\n".join(lines[:6]), username="Strategy Daily Digest", embeds=[embed])
        if ok:
            self._state.last_daily_digest_ts = self._utc_now_iso()
        else:
            self._state.notification_failures += 1
        return ok
