from __future__ import annotations

import config
from data.betfair_client import inspect_betfair_auth


def test_inspect_betfair_auth_flags_missing_key_when_only_crt_exists(tmp_path, monkeypatch):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    (cert_dir / "client-2048.crt").write_text("crt", encoding="utf-8")

    monkeypatch.setattr(config, "BF_USERNAME", "user@example.com")
    monkeypatch.setattr(config, "BF_PASSWORD", "secret")
    monkeypatch.setattr(config, "BF_APP_KEY", "app")
    monkeypatch.setattr(config, "BF_CERTS_PATH", str(cert_dir))

    payload = inspect_betfair_auth()

    assert payload["credentials_present"] is True
    assert payload["valid_cert_pair"] is False
    assert payload["primary_failure_reason"] == "cert_missing"
    assert payload["login_mode"] == "interactive"
