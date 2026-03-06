from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import config
from onchain.solana.mev_scout.latency_probe import SolanaLatencyProbe
from onchain.solana.mev_scout.provider import SolanaMevProvider
from onchain.solana.mev_scout.stream_parser import SolanaStreamParser


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MevScoutEngine:
    def __init__(self) -> None:
        self._provider = SolanaMevProvider(
            rpc_url=getattr(config, "MEV_SCOUT_SOL_RPC_URL", ""),
            ws_url=getattr(config, "MEV_SCOUT_SOL_WS_URL", ""),
            yellowstone_url=getattr(config, "MEV_SCOUT_SOL_YELLOWSTONE_URL", ""),
            replay_path=getattr(config, "MEV_SCOUT_SOL_REPLAY_PATH", ""),
        )
        self._probe = SolanaLatencyProbe()
        self._parser = SolanaStreamParser()
        self._running = False
        self._events: List[dict] = []
        self._opportunities: List[dict] = []
        self._pending: List[dict] = []
        self._latency_ms: Optional[float] = None
        self._shadow_balance = float(getattr(config, "MEV_SCOUT_SOL_SHADOW_BALANCE_USD", 5000.0))
        self._shadow_realized_pnl = 0.0
        self._balance_history: List[dict] = []
        self._label_delay_seconds = max(5, int(getattr(config, "MEV_SCOUT_SOL_LABEL_DELAY_SECONDS", 120)))
        self._min_whale_usd = float(getattr(config, "MEV_SCOUT_SOL_MIN_WHALE_USD", 250000.0))
        self._min_expected_edge_usd = float(getattr(config, "MEV_SCOUT_SOL_MIN_EXPECTED_EDGE_USD", 5.0))
        self._stats: Dict[str, float] = {
            "observed_events": 0,
            "whale_events": 0,
            "opportunities_opened": 0,
            "opportunities_settled": 0,
            "winning_opportunities": 0,
            "losing_opportunities": 0,
            "avg_expected_edge_usd": 0.0,
            "avg_realized_edge_usd": 0.0,
            "calibration_mae_usd": 0.0,
            "positive_rate": 0.0,
        }
        self._recent_realized: Deque[float] = deque(maxlen=50)
        self._recent_expected: Deque[float] = deque(maxlen=50)
        self._venue_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"observed": 0.0, "settled": 0.0, "wins": 0.0, "avg_realized_edge_usd": 0.0}
        )
        self._last_update_time: Optional[str] = None

    async def start(self) -> None:
        self._running = True
        await self._provider.start()
        while self._running:
            if self._provider.configured():
                self._latency_ms = await self._probe.probe(
                    self._provider.rpc_url or self._provider.ws_url or self._provider.yellowstone_url
                )
                self._process_raw_events(await self._provider.poll_events())
                self._settle_pending()
            self._update_balance_history()
            self._events = self._events[-300:]
            self._opportunities = self._opportunities[-300:]
            await asyncio.sleep(10.0)

    async def stop(self) -> None:
        self._running = False
        await self._provider.stop()

    def _process_raw_events(self, raw_events: List[dict]) -> None:
        now = _utc_now()
        for raw in raw_events:
            parsed = self._parser.parse(raw)
            if not parsed:
                continue
            parsed["ts"] = parsed.get("ts") or now.isoformat()
            parsed["amount_usd"] = float(parsed.get("amount_usd", 0.0) or 0.0)
            parsed["latency_ms"] = float(parsed.get("latency_hint_ms", self._latency_ms or 0.0) or 0.0)
            self._events.append(parsed)
            self._stats["observed_events"] += 1
            self._venue_stats[parsed["venue"]]["observed"] += 1
            if parsed["amount_usd"] < self._min_whale_usd:
                continue
            self._stats["whale_events"] += 1
            expected_edge_usd = self._expected_edge_usd(parsed)
            if expected_edge_usd < self._min_expected_edge_usd:
                continue
            opp = {
                "opportunity_id": parsed.get("signature") or f"opp-{len(self._opportunities) + len(self._pending) + 1}",
                "signature": parsed.get("signature"),
                "wallet": parsed.get("wallet"),
                "venue": parsed.get("venue"),
                "route": parsed.get("route"),
                "route_hops": int(parsed.get("route_hops", 1) or 1),
                "amount_usd": parsed["amount_usd"],
                "latency_ms": parsed["latency_ms"],
                "opened_at": now.isoformat(),
                "expected_edge_usd": round(expected_edge_usd, 6),
                "status": "OPEN",
                "label_source": "heuristic" if parsed.get("realized_edge_usd") is None else "observed",
                "realized_edge_usd": None,
            }
            self._pending.append({**opp, "_parsed": parsed, "_opened_dt": now})
            self._opportunities.append(dict(opp))
            self._stats["opportunities_opened"] += 1
            self._recent_expected.append(expected_edge_usd)

    def _expected_edge_usd(self, event: Dict[str, float]) -> float:
        amount = float(event.get("amount_usd", 0.0) or 0.0)
        latency = float(event.get("latency_ms", self._latency_ms or 0.0) or 0.0)
        route_hops = max(1, int(event.get("route_hops", 1) or 1))
        venue = str(event.get("venue", "unknown"))
        venue_avg = float(self._venue_stats[venue]["avg_realized_edge_usd"] or 0.0)
        base_edge = amount * 0.00012
        latency_factor = max(0.15, 1.0 - (latency / 1200.0))
        route_penalty = 1.0 / route_hops
        prior_boost = 1.0 + min(0.5, max(-0.5, venue_avg / 100.0))
        return max(0.0, base_edge * latency_factor * route_penalty * prior_boost)

    def _realized_edge_usd(self, pending: Dict[str, object]) -> float:
        parsed = dict(pending.get("_parsed") or {})
        explicit = parsed.get("realized_edge_usd")
        if explicit is not None:
            try:
                return float(explicit)
            except Exception:
                pass
        realized_bps = parsed.get("realized_edge_bps")
        if realized_bps is not None:
            try:
                return float(parsed.get("amount_usd", 0.0) or 0.0) * (float(realized_bps) / 10000.0)
            except Exception:
                pass
        expected = float(pending.get("expected_edge_usd", 0.0) or 0.0)
        latency = float(parsed.get("latency_ms", self._latency_ms or 0.0) or 0.0)
        venue = str(parsed.get("venue", "unknown"))
        venue_bias = {"jupiter": 1.1, "raydium": 1.0, "orca": 0.95}.get(venue, 0.9)
        latency_drag = max(0.1, 1.0 - (latency / 900.0))
        amount_scale = min(1.5, max(0.8, float(parsed.get("amount_usd", 0.0) or 0.0) / max(self._min_whale_usd, 1.0)))
        return round(expected * venue_bias * latency_drag * amount_scale, 6)

    def _settle_pending(self) -> None:
        now = _utc_now()
        settled: List[dict] = []
        still_open: List[dict] = []
        for pending in self._pending:
            opened_dt = pending.get("_opened_dt")
            if not isinstance(opened_dt, datetime):
                still_open.append(pending)
                continue
            if (now - opened_dt).total_seconds() < self._label_delay_seconds and pending.get("_parsed", {}).get("realized_edge_usd") is None:
                still_open.append(pending)
                continue
            realized_edge = self._realized_edge_usd(pending)
            pending["closed_at"] = now.isoformat()
            pending["realized_edge_usd"] = round(realized_edge, 6)
            pending["status"] = "CLOSED"
            pending["close_reason"] = "labeled"
            settled.append(pending)
        self._pending = still_open
        if not settled:
            return
        for opp in settled:
            self._shadow_realized_pnl += float(opp["realized_edge_usd"])
            self._stats["opportunities_settled"] += 1
            if float(opp["realized_edge_usd"]) >= 0:
                self._stats["winning_opportunities"] += 1
            else:
                self._stats["losing_opportunities"] += 1
            self._recent_realized.append(float(opp["realized_edge_usd"]))
            venue = str(opp.get("venue", "unknown"))
            venue_stat = self._venue_stats[venue]
            venue_stat["settled"] += 1
            if float(opp["realized_edge_usd"]) >= 0:
                venue_stat["wins"] += 1
            prev_avg = float(venue_stat["avg_realized_edge_usd"] or 0.0)
            venue_stat["avg_realized_edge_usd"] = prev_avg + (
                (float(opp["realized_edge_usd"]) - prev_avg) / max(1.0, venue_stat["settled"])
            )
            for item in self._opportunities:
                if item.get("opportunity_id") == opp.get("opportunity_id"):
                    item.update(
                        {
                            "status": "CLOSED",
                            "closed_at": opp["closed_at"],
                            "realized_edge_usd": opp["realized_edge_usd"],
                            "close_reason": "labeled",
                        }
                    )
                    break
        self._update_stats()
        self._last_update_time = now.isoformat()

    def _update_stats(self) -> None:
        settled = int(self._stats["opportunities_settled"])
        if self._recent_expected:
            self._stats["avg_expected_edge_usd"] = round(sum(self._recent_expected) / len(self._recent_expected), 6)
        if self._recent_realized:
            self._stats["avg_realized_edge_usd"] = round(sum(self._recent_realized) / len(self._recent_realized), 6)
        if settled:
            self._stats["positive_rate"] = round(self._stats["winning_opportunities"] / settled, 4)
        pair_count = min(len(self._recent_expected), len(self._recent_realized))
        if pair_count:
            total_abs_error = 0.0
            exp = list(self._recent_expected)[-pair_count:]
            real = list(self._recent_realized)[-pair_count:]
            for expected, realized in zip(exp, real):
                total_abs_error += abs(expected - realized)
            self._stats["calibration_mae_usd"] = round(total_abs_error / pair_count, 6)

    def _update_balance_history(self) -> None:
        balance = self._shadow_balance + self._shadow_realized_pnl
        if not self._balance_history or abs(float(self._balance_history[-1].get("balance", 0.0)) - balance) > 1e-9:
            self._balance_history.append({"ts": _utc_now().isoformat(), "balance": round(balance, 6)})

    def _readiness(self) -> Dict[str, object]:
        if not self._provider.configured():
            return {
                "status": "research_only",
                "research_only": True,
                "blockers": ["missing_rpc_or_replay_configuration"],
                "confidence": "low",
            }
        settled = int(self._stats["opportunities_settled"])
        blockers: List[str] = []
        status = "paper_validating"
        if settled < 20:
            blockers.append("insufficient_labeled_opportunities")
        if self._stats["positive_rate"] < 0.55 and settled >= 20:
            blockers.append("positive_rate_below_threshold")
        if self._stats["avg_realized_edge_usd"] <= 0 and settled >= 20:
            blockers.append("avg_realized_edge_non_positive")
        if blockers:
            status = "paper_validating"
        elif settled >= 20:
            status = "candidate"
        checks = [
            {"name": "provider_configured", "ok": True, "reason": "RPC/WS or replay source must be configured"},
            {"name": "labeled_opportunities", "ok": settled >= 20, "reason": "Need at least 20 labeled opportunities"},
            {"name": "positive_rate", "ok": settled < 20 or self._stats["positive_rate"] >= 0.55, "reason": "Positive opportunity rate should exceed 55%"},
            {"name": "avg_realized_edge", "ok": settled < 20 or self._stats["avg_realized_edge_usd"] > 0, "reason": "Average realized shadow edge should remain positive"},
        ]
        return {
            "status": status,
            "research_only": False,
            "blockers": blockers,
            "confidence": "medium" if settled >= 20 else "low",
            "score_pct": round((sum(1 for item in checks if item["ok"]) / len(checks)) * 100.0, 2),
            "checks": checks,
        }

    def _top_venues(self) -> List[Dict[str, object]]:
        ranked = sorted(
            self._venue_stats.items(),
            key=lambda item: (item[1]["avg_realized_edge_usd"], item[1]["settled"]),
            reverse=True,
        )
        out: List[Dict[str, object]] = []
        for venue, stats in ranked[:5]:
            out.append(
                {
                    "venue": venue,
                    "observed": int(stats["observed"]),
                    "settled": int(stats["settled"]),
                    "wins": int(stats["wins"]),
                    "avg_realized_edge_usd": round(float(stats["avg_realized_edge_usd"]), 6),
                }
            )
        return out

    def _learner_metrics(self) -> Dict[str, object]:
        settled = int(self._stats["opportunities_settled"])
        strict_gate_pass = settled >= 20 and self._stats["positive_rate"] >= 0.55 and self._stats["avg_realized_edge_usd"] > 0
        return {
            "running": self._running,
            "provider_configured": self._provider.configured(),
            "provider_mode": self._provider.mode(),
            "observed_events": int(self._stats["observed_events"]),
            "whale_events": int(self._stats["whale_events"]),
            "opportunities_opened": int(self._stats["opportunities_opened"]),
            "settled_count": settled,
            "win_rate": round(self._stats["positive_rate"] * 100.0, 2),
            "avg_expected_edge_usd": self._stats["avg_expected_edge_usd"],
            "avg_realized_edge_usd": self._stats["avg_realized_edge_usd"],
            "calibration_mae_usd": self._stats["calibration_mae_usd"],
            "last_update_time": self._last_update_time,
            "strict_gate_pass": strict_gate_pass,
            "strict_gate_reason": "" if strict_gate_pass else ("insufficient_labeled_opportunities" if settled < 20 else "edge_quality"),
            "rolling_50": {
                "settled": settled,
                "avg_realized_edge_usd": self._stats["avg_realized_edge_usd"],
                "win_rate": round(self._stats["positive_rate"] * 100.0, 2),
            },
        }

    def get_state(self) -> dict:
        readiness = self._readiness()
        return {
            "portfolio_id": "mev_scout_sol",
            "running": self._running,
            "mode": "research_only",
            "status": "running" if self._running else "idle",
            "shadow_balance_usd": round(self._shadow_balance + self._shadow_realized_pnl, 6),
            "shadow_realized_pnl_usd": round(self._shadow_realized_pnl, 6),
            "opportunity_count": len(self._opportunities),
            "open_opportunity_count": len(self._pending),
            "latency_ms": self._latency_ms,
            "events": self._events[-100:],
            "opportunities": self._opportunities[-100:],
            "balance_history": self._balance_history[-1000:],
            "readiness": readiness,
            "provider_configured": self._provider.configured(),
            "provider_mode": self._provider.mode(),
            "learner": self._learner_metrics(),
            "venue_stats": self._top_venues(),
            "training_progress": {
                "observed_events": int(self._stats["observed_events"]),
                "whale_events": int(self._stats["whale_events"]),
                "labeled_opportunities": int(self._stats["opportunities_settled"]),
                "min_labeled_target": 20,
            },
        }
