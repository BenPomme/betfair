from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import httpx

import config


_SPORT_MAP = {
    "soccer": "Soccer",
    "football": "Soccer",
    "tennis": "Tennis",
    "basketball": "Basketball",
    "baseball": "Baseball",
    "hockey": "Ice Hockey",
    "mma": "MMA",
    "boxing": "Boxing",
    "golf": "Golf",
    "cricket": "Cricket",
    "rugby": "Rugby",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _build_title(home: str, away: str, fallback: str) -> str:
    if home and away:
        return f"{home} vs {away}"
    return fallback


class TheSportsDBAdapter:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        lookahead_days: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or config.THESPORTSDB_HTTP_BASE_URL).rstrip("/")
        self.api_key = str(api_key or config.THESPORTSDB_API_KEY).strip()
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else config.BETFAIR_EXTERNAL_SOURCE_HTTP_TIMEOUT_SECONDS
        )
        self.lookahead_days = int(lookahead_days if lookahead_days is not None else config.THESPORTSDB_LOOKAHEAD_DAYS)

    async def _get_json(self, client: httpx.AsyncClient, date_str: str, sport: str) -> Any:
        path = f"/api/v1/json/{self.api_key}/eventsday.php"
        response = await client.get(f"{self.base_url}{path}", params={"d": date_str, "s": sport})
        response.raise_for_status()
        return response.json()

    def _normalize_event(self, row: Dict[str, Any], sport: str) -> Dict[str, Any]:
        home = str(row.get("strHomeTeam") or "").strip()
        away = str(row.get("strAwayTeam") or "").strip()
        fallback_title = str(row.get("strEvent") or row.get("strFilename") or row.get("idEvent") or "").strip()
        title = _build_title(home, away, fallback_title)
        event_date = str(row.get("dateEvent") or "").strip()
        event_time = str(row.get("strTime") or "").strip()
        start_time = _parse_dt(f"{event_date}T{event_time}Z" if event_date and event_time else row.get("strTimestamp"))
        teams_or_players = [name for name in (home, away) if name]
        return {
            "source": "thesportsdb",
            "sport": sport.lower(),
            "event_key": str(row.get("idEvent") or title),
            "event_type": "score_event_confirmation",
            "title": title,
            "participants": teams_or_players,
            "scheduled_start": start_time.isoformat() if start_time else None,
            "observed_at": _utc_now_iso(),
            "confidence": 0.68,
            "competition": str(row.get("strLeague") or row.get("strLeagueAlternate") or "").strip(),
            "status": str(row.get("strStatus") or "").strip(),
            "raw_payload": dict(row),
        }

    async def fetch_snapshot(self, allowed_sports: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        sports = sorted(
            {
                _SPORT_MAP.get(str(sport).strip().lower())
                for sport in (allowed_sports or _SPORT_MAP.keys())
                if _SPORT_MAP.get(str(sport).strip().lower())
            }
        )
        if not sports:
            sports = ["Soccer", "Tennis"]

        observed_at = _utc_now_iso()
        rows: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for offset in range(max(1, self.lookahead_days)):
                date_str = (_utc_now() + timedelta(days=offset)).date().isoformat()
                for sport in sports:
                    payload = await self._get_json(client, date_str, sport)
                    for row in payload.get("events") or []:
                        if isinstance(row, dict):
                            rows.append(self._normalize_event(row, sport))
        return {
            "provider": "thesportsdb",
            "observed_at": observed_at,
            "events": rows,
            "quotes": [],
            "sports": sorted({str(row.get("sport") or "") for row in rows if row.get("sport")}),
            "event_count": len(rows),
            "quote_count": 0,
            "healthy": bool(rows),
            "http_base_url": self.base_url,
            "feed_health": "healthy" if rows else "degraded",
        }
