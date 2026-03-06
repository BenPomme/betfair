from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List

import config
from onchain.solana.mev_scout.latency_probe import SolanaLatencyProbe
from onchain.solana.mev_scout.provider import SolanaMevProvider
from onchain.solana.mev_scout.stream_parser import SolanaStreamParser


class MevScoutEngine:
    def __init__(self) -> None:
        self._provider = SolanaMevProvider(
            rpc_url=getattr(config, "MEV_SCOUT_SOL_RPC_URL", ""),
            ws_url=getattr(config, "MEV_SCOUT_SOL_WS_URL", ""),
            yellowstone_url=getattr(config, "MEV_SCOUT_SOL_YELLOWSTONE_URL", ""),
        )
        self._probe = SolanaLatencyProbe()
        self._parser = SolanaStreamParser()
        self._running = False
        self._events: List[dict] = []
        self._opportunities: List[dict] = []
        self._latency_ms = None
        self._shadow_balance = float(getattr(config, "MEV_SCOUT_SOL_SHADOW_BALANCE_USD", 5000.0))
        self._shadow_realized_pnl = 0.0
        self._balance_history: List[dict] = []

    async def start(self) -> None:
        self._running = True
        await self._provider.start()
        while self._running:
            if self._provider.configured():
                self._latency_ms = await self._probe.probe(self._provider.rpc_url or self._provider.ws_url or self._provider.yellowstone_url)
                for raw in await self._provider.poll_events():
                    parsed = self._parser.parse(raw)
                    if parsed:
                        parsed["ts"] = datetime.now(timezone.utc).isoformat()
                        self._events.append(parsed)
                        if float(parsed.get("amount_usd", 0.0) or 0.0) >= float(getattr(config, "MEV_SCOUT_SOL_MIN_WHALE_USD", 250000.0)):
                            opp = dict(parsed)
                            opp["theoretical_net_edge_usd"] = round(float(parsed.get("amount_usd", 0.0)) * 0.00015, 6)
                            self._opportunities.append(opp)
                            self._shadow_realized_pnl += float(opp["theoretical_net_edge_usd"])
            balance = self._shadow_balance + self._shadow_realized_pnl
            if not self._balance_history or abs(float(self._balance_history[-1].get("balance", 0.0)) - balance) > 1e-9:
                self._balance_history.append({"ts": datetime.now(timezone.utc).isoformat(), "balance": round(balance, 6)})
            self._events = self._events[-300:]
            self._opportunities = self._opportunities[-300:]
            await asyncio.sleep(10.0)

    async def stop(self) -> None:
        self._running = False
        await self._provider.stop()

    def get_state(self) -> dict:
        readiness = {
            "status": "research_only",
            "research_only": True,
            "blockers": [] if self._provider.configured() else ["missing_rpc_configuration"],
            "confidence": "low",
        }
        return {
            "portfolio_id": "mev_scout_sol",
            "running": self._running,
            "mode": "research_only",
            "status": "running" if self._running else "idle",
            "shadow_balance_usd": round(self._shadow_balance + self._shadow_realized_pnl, 6),
            "shadow_realized_pnl_usd": round(self._shadow_realized_pnl, 6),
            "opportunity_count": len(self._opportunities),
            "latency_ms": self._latency_ms,
            "events": self._events[-100:],
            "opportunities": self._opportunities[-100:],
            "balance_history": self._balance_history[-1000:],
            "readiness": readiness,
            "provider_configured": self._provider.configured(),
        }
