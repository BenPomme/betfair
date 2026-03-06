import json
from decimal import Decimal
from pathlib import Path

import config
from funding.core.schemas import HedgePosition
from funding.execution.position_manager import PositionManager


def test_begin_validation_run_archives_existing_state(tmp_path, monkeypatch):
    state_path = tmp_path / "funding_positions.json"
    archive_dir = tmp_path / "archive"
    monkeypatch.setattr(config, "FUNDING_VALIDATION_ARCHIVE_DIR", str(archive_dir))
    state_path.write_text(
        json.dumps(
            {
                "positions": [
                    HedgePosition(symbol="BTCUSDT", quantity_spot=Decimal("1"), entry_price_spot=Decimal("100")).to_dict()
                ],
                "last_updated": "2026-03-06T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pm = PositionManager(state_path=str(state_path))
    archived = pm.begin_validation_run("run_123", manifest={"scope": "hedge_only"})

    assert archived is not None
    assert archived.exists()
    manifest = json.loads((archived.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run_123"
    current = json.loads(state_path.read_text(encoding="utf-8"))
    assert current["positions"] == []
    assert current["validation_run_id"] == "run_123"


def test_from_dict_backward_compat_loads_old_position():
    raw = {
        "id": "abc",
        "symbol": "ETHUSDT",
        "entry_price_spot": "100",
        "entry_price_perp": "101",
        "quantity_spot": "1",
        "quantity_perp": "1",
        "leverage": 2,
        "margin_type": "ISOLATED",
        "entry_time": None,
        "funding_collected": "0",
        "trading_fees_paid": "0",
        "status": "OPEN",
        "exit_time": None,
        "exit_price_spot": "0",
        "exit_price_perp": "0",
        "exit_pnl": "0",
    }
    position = HedgePosition.from_dict(raw)
    assert position.validation_run_id == ""
    assert position.realized_funding_events == 0
