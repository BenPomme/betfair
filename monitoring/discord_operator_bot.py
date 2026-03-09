from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import discord
import httpx

import config

logger = logging.getLogger(__name__)


def _parse_id_set(raw: str) -> Set[int]:
    parsed: Set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.add(int(token))
        except ValueError:
            logger.warning("Ignoring invalid Discord ID token: %s", token)
    return parsed


class CommandCenterClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self.http.aclose()

    async def get_json(self, path: str) -> Dict[str, Any]:
        response = await self.http.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    async def post_json(self, path: str) -> Dict[str, Any]:
        response = await self.http.post(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()


class OperatorBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.prefix = (str(config.DISCORD_BOT_PREFIX or "!").strip() or "!")
        self.allowed_users = _parse_id_set(config.DISCORD_BOT_ALLOWED_USER_IDS)
        self.allowed_guilds = _parse_id_set(config.DISCORD_BOT_ALLOWED_GUILD_IDS)
        self.allowed_channels = _parse_id_set(config.DISCORD_BOT_ALLOWED_CHANNEL_IDS)
        self.allow_dms = bool(getattr(config, "DISCORD_BOT_ALLOW_DMS", True))
        self.inbox_path = Path(
            str(getattr(config, "DISCORD_BOT_INBOX_PATH", "data/runtime/discord_operator_inbox.jsonl"))
        )
        self.status_portfolios = [
            token.strip()
            for token in str(config.DISCORD_BOT_STATUS_PORTFOLIOS or "").split(",")
            if token.strip()
        ]
        self.command_center = CommandCenterClient(
            f"http://127.0.0.1:{int(config.COMMAND_CENTER_PORT)}"
        )

    async def close(self) -> None:
        await self.command_center.close()
        await super().close()

    async def on_ready(self) -> None:
        logger.info(
            "Discord operator bot ready user=%s allowed_users=%s allowed_guilds=%s allowed_channels=%s",
            self.user,
            sorted(self.allowed_users),
            sorted(self.allowed_guilds),
            sorted(self.allowed_channels),
        )

    def _authorized(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        if self.allowed_users and int(message.author.id) not in self.allowed_users:
            return False
        guild = getattr(message, "guild", None)
        if guild is None:
            return bool(self.allowed_users) and self.allow_dms
        if self.allowed_guilds:
            if int(guild.id) not in self.allowed_guilds:
                return False
        if self.allowed_channels and int(message.channel.id) not in self.allowed_channels:
            return False
        if not self.allowed_users:
            return False
        return True

    async def on_message(self, message: discord.Message) -> None:
        if not self._authorized(message):
            return
        content = (message.content or "").strip()
        if getattr(message, "guild", None) is None and content and not content.startswith(self.prefix):
            reply = await self._store_inbox_message(message)
            if reply:
                await message.reply(reply[:1900], mention_author=False)
            return
        if not content.startswith(self.prefix):
            return
        command = content[len(self.prefix) :].strip()
        if not command:
            return
        try:
            reply = await self.handle_command(command)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Discord command failed")
            reply = f"error: {exc}"
        if reply:
            await message.reply(reply[:1900], mention_author=False)

    async def handle_command(self, command: str) -> str:
        parts = command.split()
        if not parts:
            return self._help_text()
        verb = parts[0].lower()
        if verb in {"help", "commands"}:
            return self._help_text()
        if verb == "status":
            return await self._status_text()
        if verb == "inbox":
            return self._inbox_text()
        if verb in {"portfolio", "pnl", "blockers"}:
            if len(parts) < 2:
                return f"usage: `{self.prefix}{verb} <portfolio_id>`"
            return await self._portfolio_text(parts[1], mode=verb)
        if verb in {"start", "stop", "restart"}:
            if len(parts) < 2:
                return f"usage: `{self.prefix}{verb} <portfolio_id>`"
            return await self._control_text(verb, parts[1])
        return f"unknown command. use `{self.prefix}help`"

    def _help_text(self) -> str:
        return (
            "allowed commands:\n"
            f"- `{self.prefix}status`\n"
            f"- `{self.prefix}inbox`\n"
            f"- `{self.prefix}portfolio <portfolio_id>`\n"
            f"- `{self.prefix}pnl <portfolio_id>`\n"
            f"- `{self.prefix}blockers <portfolio_id>`\n"
            f"- `{self.prefix}start <portfolio_id>`\n"
            f"- `{self.prefix}stop <portfolio_id>`\n"
            f"- `{self.prefix}restart <portfolio_id>`\n"
            f"- DM any non-command message to queue it for operator review"
        )

    async def _store_inbox_message(self, message: discord.Message) -> str:
        attachments = [
            {
                "filename": item.filename,
                "content_type": item.content_type,
                "size": item.size,
                "url": item.url,
            }
            for item in list(message.attachments or [])
        ]
        entry = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "author_id": int(message.author.id),
            "author_name": str(message.author),
            "channel_id": int(message.channel.id),
            "channel_type": type(message.channel).__name__,
            "content": str(message.content or ""),
            "attachments": attachments,
        }
        self.inbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.inbox_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        return (
            "Message stored for operator review.\n"
            f"Queue file: `{self.inbox_path.as_posix()}`\n"
            f"Use `{self.prefix}inbox` to confirm the queue depth."
        )

    def _inbox_text(self) -> str:
        if not self.inbox_path.exists():
            return "operator inbox is empty"
        lines = self.inbox_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return "operator inbox is empty"
        entries: List[Dict[str, Any]] = []
        for line in lines[-5:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        reply = [f"operator inbox: {len(lines)} queued message(s)"]
        for item in entries:
            preview = str(item.get("content") or "").strip().replace("\n", " ")
            preview = (preview[:90] + "...") if len(preview) > 93 else (preview or "<attachment-only>")
            reply.append(
                f"- {item.get('received_at', 'unknown')} | {item.get('author_name', 'unknown')} | {preview}"
            )
        return "\n".join(reply[:6])

    async def _status_text(self) -> str:
        payload = await self.command_center.get_json("/api/portfolios")
        portfolios = payload.get("portfolios") or []
        lines: List[str] = ["portfolio status"]
        wanted = set(self.status_portfolios)
        for item in portfolios:
            portfolio_id = str(item.get("portfolio_id") or "")
            if wanted and portfolio_id not in wanted:
                continue
            lines.append(
                f"- {portfolio_id}: {item.get('readiness')} | "
                f"PnL {float(item.get('realized_pnl', 0.0) or 0.0):+.2f} {item.get('currency', '')} | "
                f"progress {float(item.get('progress_pct', 0.0) or 0.0):.0f}% | running={bool(item.get('running'))}"
            )
        return "\n".join(lines[:20])

    async def _portfolio_text(self, portfolio_id: str, *, mode: str) -> str:
        state = await self.command_center.get_json(f"/api/portfolios/{portfolio_id}/state")
        summary = await self.command_center.get_json(f"/api/portfolios/{portfolio_id}/summary")
        account = state.get("account") or {}
        readiness = state.get("readiness") or {}
        blockers = readiness.get("blockers") or []
        if mode == "pnl":
            return (
                f"{portfolio_id} PnL\n"
                f"realized: {float(account.get('realized_pnl', 0.0) or 0.0):+.2f} {account.get('currency', summary.get('currency', ''))}\n"
                f"unrealized: {float(account.get('unrealized_pnl', 0.0) or 0.0):+.2f}\n"
                f"roi: {float(account.get('roi_pct', 0.0) or 0.0):+.2f}%\n"
                f"trades: {int(account.get('trade_count', 0) or 0)}"
            )
        if mode == "blockers":
            if not blockers:
                return f"{portfolio_id} blockers: none"
            return f"{portfolio_id} blockers:\n- " + "\n- ".join(map(str, blockers[:8]))
        return (
            f"{portfolio_id}\n"
            f"status: {summary.get('status')} | readiness: {summary.get('readiness')}\n"
            f"progress: {float(summary.get('progress_pct', 0.0) or 0.0):.0f}% | eta: {summary.get('eta_to_readiness')}\n"
            f"pnl: {float(account.get('realized_pnl', 0.0) or 0.0):+.2f} {account.get('currency', summary.get('currency', ''))}\n"
            f"open: {int(summary.get('open_count', 0) or 0)} | blockers: {len(blockers)}"
        )

    async def _control_text(self, action: str, portfolio_id: str) -> str:
        result = await self.command_center.post_json(f"/api/portfolios/{portfolio_id}/{action}")
        return f"{action} {portfolio_id}: ok={bool(result.get('ok'))} pid={result.get('pid')}"


def bot_configured() -> bool:
    return bool(config.DISCORD_BOT_ENABLED and config.DISCORD_BOT_TOKEN)


async def run_bot() -> None:
    if not bot_configured():
        raise RuntimeError("Discord bot not configured")
    bot = OperatorBot()
    await bot.start(config.DISCORD_BOT_TOKEN)


def main() -> None:
    if not bot_configured():
        print("discord operator bot disabled or missing token")
        return
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
