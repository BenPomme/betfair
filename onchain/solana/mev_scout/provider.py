from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class SolanaMevProvider:
    def __init__(self, rpc_url: str = "", ws_url: str = "", yellowstone_url: str = "") -> None:
        self.rpc_url = rpc_url
        self.ws_url = ws_url
        self.yellowstone_url = yellowstone_url
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def poll_events(self) -> List[Dict[str, Any]]:
        if not self._running:
            return []
        await asyncio.sleep(0)
        return []

    def configured(self) -> bool:
        return bool(self.rpc_url or self.ws_url or self.yellowstone_url)
