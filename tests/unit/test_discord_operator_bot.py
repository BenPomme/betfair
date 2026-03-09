import asyncio
import json
from types import SimpleNamespace

import config
from monitoring.discord_operator_bot import OperatorBot


def _make_message(*, author_id=123, guild_id=None, channel_id=456, content="hello"):
    guild = None if guild_id is None else SimpleNamespace(id=guild_id)
    channel = SimpleNamespace(id=channel_id)
    author = SimpleNamespace(id=author_id, bot=False)
    author.__str__ = lambda self=author: "tester#0001"
    return SimpleNamespace(
        author=author,
        guild=guild,
        channel=channel,
        content=content,
        attachments=[],
    )


def test_operator_bot_authorizes_allowed_direct_messages(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOWED_USER_IDS", "123")
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOWED_GUILD_IDS", "999")
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOWED_CHANNEL_IDS", "")
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOW_DMS", True)

    bot = OperatorBot()
    try:
        assert bot._authorized(_make_message()) is True
        assert bot._authorized(_make_message(author_id=999)) is False
        assert bot._authorized(_make_message(guild_id=999)) is True
    finally:
        asyncio.run(bot.close())


def test_operator_bot_stores_direct_messages_in_inbox(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOWED_USER_IDS", "123")
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOWED_GUILD_IDS", "")
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOWED_CHANNEL_IDS", "")
    monkeypatch.setattr(config, "DISCORD_BOT_ALLOW_DMS", True)
    monkeypatch.setattr(config, "DISCORD_BOT_INBOX_PATH", str(tmp_path / "discord_operator_inbox.jsonl"))

    bot = OperatorBot()
    try:
        message = _make_message(content="binance api key incoming")
        reply = asyncio.run(bot._store_inbox_message(message))
        assert "Message stored for operator review." in reply

        lines = (tmp_path / "discord_operator_inbox.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["author_id"] == 123
        assert payload["content"] == "binance api key incoming"
    finally:
        asyncio.run(bot.close())
