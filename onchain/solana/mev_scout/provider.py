from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


class SolanaMevProvider:
    def __init__(
        self,
        rpc_url: str = "",
        ws_url: str = "",
        yellowstone_url: str = "",
        replay_path: str = "",
    ) -> None:
        self.rpc_url = rpc_url
        self.ws_url = ws_url
        self.yellowstone_url = yellowstone_url
        self.replay_path = replay_path or getattr(config, "MEV_SCOUT_SOL_REPLAY_PATH", "")
        self._running = False
        self._replay_events: List[Dict[str, Any]] = []
        self._replay_index = 0

    async def start(self) -> None:
        self._running = True
        self._load_replay()

    async def stop(self) -> None:
        self._running = False

    async def poll_events(self) -> List[Dict[str, Any]]:
        if not self._running:
            return []
        if self._replay_events:
            limit = max(1, int(getattr(config, "MEV_SCOUT_SOL_MAX_EVENTS_PER_POLL", 25)))
            batch = self._replay_events[self._replay_index : self._replay_index + limit]
            self._replay_index = min(len(self._replay_events), self._replay_index + len(batch))
            await asyncio.sleep(0)
            return batch
        await asyncio.sleep(0)
        return []

    def configured(self) -> bool:
        return bool(self.rpc_url or self.ws_url or self.yellowstone_url or self._resolved_replay_path())

    def mode(self) -> str:
        if self._replay_events:
            return "replay"
        if self.rpc_url or self.ws_url or self.yellowstone_url:
            return "live"
        return "none"

    def _resolved_replay_path(self) -> Optional[Path]:
        if not self.replay_path:
            return None
        path = Path(self.replay_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _load_replay(self) -> None:
        self._replay_events = []
        self._replay_index = 0
        path = self._resolved_replay_path()
        if path is None or not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                self._replay_events.append(payload)
