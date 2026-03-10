from __future__ import annotations

import asyncio
import sys
import types

import httpx

discord_stub = types.ModuleType("discord")


class _StubIntents:
    def __init__(self) -> None:
        self.message_content = False

    @classmethod
    def default(cls):
        return cls()


class _StubClient:
    def __init__(self, *, intents=None) -> None:
        self.intents = intents

    async def close(self) -> None:
        return None


discord_stub.Intents = _StubIntents
discord_stub.Client = _StubClient
discord_stub.Message = object
sys.modules.setdefault("discord", discord_stub)

import config
from monitoring.discord_operator_bot import CommandCenterClient, CommandCenterError, OperatorBot


class _HealthyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def close(self) -> None:
        return None

    async def get_json(self, path: str):
        assert path == "/api/portfolios"
        return {
            "portfolios": [
                {"portfolio_id": "hedge_validation", "running": True},
                {"portfolio_id": "hedge_research", "running": False},
            ]
        }

    async def post_json(self, path: str):
        return {"ok": True}


class _RestartFailureClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def close(self) -> None:
        return None

    async def get_json(self, path: str):
        return {"portfolios": []}

    async def post_json(self, path: str):
        raise CommandCenterError("command center http 400: already_running (pid=4321)")


def test_command_center_client_formats_http_detail_error():
    client = CommandCenterClient("http://127.0.0.1:8000")
    response = httpx.Response(
        400,
        json={"detail": {"error": "already_running", "pid": 4321}},
    )

    try:
        client._raise_for_status(response)
    except CommandCenterError as exc:
        assert str(exc) == "command center http 400: already_running (pid=4321)"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected CommandCenterError")
    finally:
        asyncio.run(client.close())


def test_operator_bot_health_command(monkeypatch):
    monkeypatch.setattr(config, "COMMAND_CENTER_PORT", 8011)
    monkeypatch.setattr("monitoring.discord_operator_bot.CommandCenterClient", _HealthyClient)

    async def _run() -> None:
        bot = OperatorBot()
        try:
            reply = await bot.handle_command("health")
            assert reply == (
                "health: command center reachable | portfolios=2 | running=1 "
                "| host=127.0.0.1:8011"
            )
        finally:
            await bot.close()

    asyncio.run(_run())


def test_operator_bot_restart_failure_returns_actionable_error(monkeypatch):
    monkeypatch.setattr("monitoring.discord_operator_bot.CommandCenterClient", _RestartFailureClient)

    async def _run() -> None:
        bot = OperatorBot()
        try:
            try:
                await bot.handle_command("restart hedge_validation")
            except Exception as exc:
                reply = bot._format_exception(exc)
            else:  # pragma: no cover - defensive
                raise AssertionError("expected restart to fail")
            assert reply == "error: command center http 400: already_running (pid=4321)"
        finally:
            await bot.close()

    asyncio.run(_run())
