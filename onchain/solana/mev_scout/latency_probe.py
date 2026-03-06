from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional


class SolanaLatencyProbe:
    async def probe(self, url: str) -> Optional[float]:
        if not url:
            return None
        started = datetime.now(timezone.utc)
        await asyncio.sleep(0.01)
        return round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 4)
