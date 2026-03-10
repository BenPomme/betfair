from __future__ import annotations

from dataclasses import dataclass

import config
from factory.binance_auth_diagnostic import format_binance_auth_diagnostics, run_binance_auth_diagnostics


@dataclass
class _FakeResponse:
    status_code: int
    payload: dict

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def reason(self) -> str:
        return "OK" if self.ok else "ERR"

    @property
    def text(self) -> str:
        return str(self.payload)

    def json(self):
        return self.payload


class _FakeSession:
    def get(self, url, headers=None, timeout=None):  # noqa: ANN001
        if "demo-fapi.binance.com" in url:
            return _FakeResponse(401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action"})
        if "fapi.binance.com" in url:
            return _FakeResponse(401, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action"})
        if "api.binance.com" in url:
            return _FakeResponse(200, {"canTrade": True, "permissions": ["SPOT"]})
        if "testnet.binance.vision" in url:
            return _FakeResponse(200, {"canTrade": True, "permissions": ["SPOT"]})
        return _FakeResponse(500, {"code": -1000, "msg": "unexpected_url"})


def test_binance_auth_diagnostic_reports_status_without_exposing_secrets(monkeypatch):
    monkeypatch.setattr(config, "BINANCE_SPOT_API_KEY", "spot-key")
    monkeypatch.setattr(config, "BINANCE_SPOT_API_SECRET", "spot-secret")
    monkeypatch.setattr(config, "BINANCE_FUTURES_API_KEY", "futures-key")
    monkeypatch.setattr(config, "BINANCE_FUTURES_API_SECRET", "futures-secret")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_API_KEY", "spot-test-key")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_API_SECRET", "spot-test-secret")
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_API_KEY", "futures-test-key")
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_API_SECRET", "futures-test-secret")
    monkeypatch.setattr(config, "BINANCE_SPOT_PROD_URL", "https://api.binance.com")
    monkeypatch.setattr(config, "BINANCE_FUTURES_PROD_URL", "https://fapi.binance.com")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_URL", "https://testnet.binance.vision")
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_URL", "https://demo-fapi.binance.com")

    results = run_binance_auth_diagnostics(session=_FakeSession())
    rendered = format_binance_auth_diagnostics(results)

    assert results["spot_prod"]["ok"] is True
    assert results["spot_testnet"]["ok"] is True
    assert results["futures_prod"]["error_code"] == -2015
    assert results["futures_testnet"]["error_code"] == -2015
    assert "spot-secret" not in rendered
    assert "futures-secret" not in rendered
