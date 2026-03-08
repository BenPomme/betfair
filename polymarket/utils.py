from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Iterable, List


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def rolling_mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        return 0.0
    return sum(rows) / len(rows)


def rolling_std(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) <= 1:
        return 0.0
    mean = rolling_mean(rows)
    variance = sum((value - mean) ** 2 for value in rows) / len(rows)
    return variance ** 0.5


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def as_millis_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        millis = int(str(value))
    except Exception:
        return None
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc).isoformat()
