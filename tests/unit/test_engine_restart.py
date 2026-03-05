import config
import monitoring.engine as engine_module
from monitoring.engine import TradingEngine


class _MainStub:
    _running = True


def test_request_runtime_restart_queues_once_and_stops_main(monkeypatch):
    monkeypatch.setattr(config, "QA_RESTART_ON_DEGRADED_ENABLED", True)
    monkeypatch.setattr(config, "QA_RESTART_COOLDOWN_SECONDS", 900)
    monkeypatch.setattr(engine_module, "_get_main", lambda: _MainStub)
    _MainStub._running = True
    engine = TradingEngine()

    first = engine._request_runtime_restart("health_degraded")
    second = engine._request_runtime_restart("health_degraded_again")

    assert first is True
    assert second is False
    assert _MainStub._running is False


def test_request_runtime_restart_respects_manual_stop(monkeypatch):
    monkeypatch.setattr(config, "QA_RESTART_ON_DEGRADED_ENABLED", True)
    monkeypatch.setattr(engine_module, "_get_main", lambda: _MainStub)
    _MainStub._running = True
    engine = TradingEngine()
    engine._manual_stop_requested = True

    queued = engine._request_runtime_restart("health_degraded")
    assert queued is False
    assert _MainStub._running is True
