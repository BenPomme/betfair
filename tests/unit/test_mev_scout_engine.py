from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import config
from onchain.solana.mev_scout.engine import MevScoutEngine


def test_mev_scout_engine_learns_from_replay(tmp_path, monkeypatch):
    replay_path = tmp_path / "mev_replay.jsonl"
    replay_rows = [
        {
            "signature": "sig-1",
            "venue": "jupiter",
            "amount_usd": 300000,
            "wallet": "wallet-a",
            "realized_edge_usd": 18.5,
            "latency_ms": 120,
        },
        {
            "signature": "sig-2",
            "venue": "raydium",
            "amount_usd": 280000,
            "wallet": "wallet-b",
            "realized_edge_usd": -3.5,
            "latency_ms": 250,
        },
    ]
    replay_path.write_text("\n".join(json.dumps(row) for row in replay_rows), encoding="utf-8")

    monkeypatch.setattr(config, "MEV_SCOUT_SOL_REPLAY_PATH", str(replay_path))
    monkeypatch.setattr(config, "MEV_SCOUT_SOL_MIN_WHALE_USD", 250000)
    monkeypatch.setattr(config, "MEV_SCOUT_SOL_MIN_EXPECTED_EDGE_USD", 1)
    monkeypatch.setattr(config, "MEV_SCOUT_SOL_MAX_EVENTS_PER_POLL", 10)
    monkeypatch.setattr(config, "MEV_SCOUT_SOL_LABEL_DELAY_SECONDS", 1)

    engine = MevScoutEngine()
    asyncio.run(engine._provider.start())
    raw_batch = asyncio.run(engine._provider.poll_events())
    engine._process_raw_events(raw_batch)

    for pending in engine._pending:
        pending["_opened_dt"] = datetime.now(timezone.utc) - timedelta(seconds=10)

    engine._settle_pending()
    state = engine.get_state()
    learner = state["learner"]

    assert state["provider_configured"] is True
    assert state["provider_mode"] == "replay"
    assert learner["observed_events"] == 2
    assert learner["whale_events"] == 2
    assert learner["settled_count"] == 2
    assert learner["avg_realized_edge_usd"] > 0
    assert state["readiness"]["status"] == "paper_validating"
