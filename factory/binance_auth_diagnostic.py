from __future__ import annotations

import hmac
import json
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable
from urllib.parse import urlencode

import requests

import config


@dataclass(frozen=True)
class _EndpointSpec:
    label: str
    base_url: str
    path: str
    api_key: str
    api_secret: str
    product: str
    environment: str


def _endpoint_specs() -> Iterable[_EndpointSpec]:
    return (
        _EndpointSpec(
            label="spot_prod",
            base_url=str(getattr(config, "BINANCE_SPOT_PROD_URL", "https://api.binance.com")),
            path="/api/v3/account",
            api_key=str(getattr(config, "BINANCE_SPOT_API_KEY", "")),
            api_secret=str(getattr(config, "BINANCE_SPOT_API_SECRET", "")),
            product="spot",
            environment="prod",
        ),
        _EndpointSpec(
            label="futures_prod",
            base_url=str(getattr(config, "BINANCE_FUTURES_PROD_URL", "https://fapi.binance.com")),
            path="/fapi/v2/account",
            api_key=str(getattr(config, "BINANCE_FUTURES_API_KEY", "")),
            api_secret=str(getattr(config, "BINANCE_FUTURES_API_SECRET", "")),
            product="futures",
            environment="prod",
        ),
        _EndpointSpec(
            label="spot_testnet",
            base_url=str(getattr(config, "BINANCE_SPOT_TESTNET_URL", "https://testnet.binance.vision")),
            path="/api/v3/account",
            api_key=str(getattr(config, "BINANCE_SPOT_TESTNET_API_KEY", "")),
            api_secret=str(getattr(config, "BINANCE_SPOT_TESTNET_API_SECRET", "")),
            product="spot",
            environment="testnet",
        ),
        _EndpointSpec(
            label="futures_testnet",
            base_url=str(getattr(config, "BINANCE_FUTURES_TESTNET_URL", "https://demo-fapi.binance.com")),
            path="/fapi/v2/account",
            api_key=str(getattr(config, "BINANCE_FUTURES_TESTNET_API_KEY", "")),
            api_secret=str(getattr(config, "BINANCE_FUTURES_TESTNET_API_SECRET", "")),
            product="futures",
            environment="testnet",
        ),
    )


def _signed_query(api_secret: str, *, timestamp_ms: int) -> str:
    query = urlencode({"timestamp": timestamp_ms, "recvWindow": 5000})
    signature = hmac.new(
        api_secret.encode("utf-8"),
        query.encode("utf-8"),
        sha256,
    ).hexdigest()
    return f"{query}&signature={signature}"


def _status_from_response(spec: _EndpointSpec, response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"message": response.text.strip()}

    status: Dict[str, Any] = {
        "label": spec.label,
        "product": spec.product,
        "environment": spec.environment,
        "base_url": spec.base_url,
        "path": spec.path,
        "ok": bool(response.ok),
        "http_status": int(response.status_code),
    }
    if response.ok:
        if isinstance(payload, dict):
            if "canTrade" in payload:
                status["can_trade"] = bool(payload.get("canTrade"))
            if "permissions" in payload:
                status["permission_count"] = len(payload.get("permissions") or [])
            if spec.product == "futures":
                status["asset_count"] = len(payload.get("assets") or [])
        return status
    if isinstance(payload, dict):
        status["error_code"] = payload.get("code")
        status["error_message"] = payload.get("msg") or payload.get("message") or response.reason
    else:
        status["error_message"] = response.reason
    return status


def check_endpoint(spec: _EndpointSpec, *, session: requests.Session | None = None) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "label": spec.label,
        "product": spec.product,
        "environment": spec.environment,
        "base_url": spec.base_url,
        "path": spec.path,
    }
    if not spec.api_key or not spec.api_secret:
        status.update({"ok": False, "error_code": "missing_credentials", "error_message": "Missing API key or secret."})
        return status

    query = _signed_query(spec.api_secret, timestamp_ms=int(time.time() * 1000))
    url = f"{spec.base_url.rstrip('/')}{spec.path}?{query}"
    client = session or requests.Session()
    try:
        response = client.get(
            url,
            headers={"X-MBX-APIKEY": spec.api_key},
            timeout=10,
        )
    except requests.RequestException as exc:
        status.update({"ok": False, "error_code": "request_failed", "error_message": str(exc)})
        return status
    return _status_from_response(spec, response)


def run_binance_auth_diagnostics(*, session: requests.Session | None = None) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for spec in _endpoint_specs():
        results[spec.label] = check_endpoint(spec, session=session)
    return results


def format_binance_auth_diagnostics(results: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for label in ("spot_prod", "futures_prod", "spot_testnet", "futures_testnet"):
        item = dict(results.get(label) or {})
        if not item:
            continue
        if item.get("ok"):
            extras = []
            if "can_trade" in item:
                extras.append(f"can_trade={item['can_trade']}")
            if "permission_count" in item:
                extras.append(f"permission_count={item['permission_count']}")
            if "asset_count" in item:
                extras.append(f"asset_count={item['asset_count']}")
            suffix = f" | {' | '.join(extras)}" if extras else ""
            rows.append(f"{label}: ok (HTTP {item.get('http_status')}){suffix}")
            continue
        message = item.get("error_message") or "unknown_error"
        code = item.get("error_code")
        if code is not None:
            rows.append(f"{label}: failed ({code}) {message}")
        else:
            rows.append(f"{label}: failed {message}")
    return "\n".join(rows)


def diagnostics_to_json(results: Dict[str, Dict[str, Any]]) -> str:
    return json.dumps(results, indent=2, sort_keys=True)
