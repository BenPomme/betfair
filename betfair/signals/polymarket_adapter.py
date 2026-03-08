from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import httpx

import config


_KNOWN_SPORT_TAGS = {
    "soccer",
    "football",
    "basketball",
    "baseball",
    "tennis",
    "hockey",
    "mma",
    "ufc",
    "golf",
    "cricket",
    "boxing",
    "f1",
    "formula-1",
    "motogp",
    "nfl",
    "nba",
    "mlb",
    "nhl",
    "wnba",
    "atp",
    "wta",
    "lal",
    "epl",
    "bundesliga",
    "serie-a",
    "ligue-1",
    "cba",
    "ncaab",
    "cfb",
    "college-basketball",
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


@dataclass(frozen=True)
class PolymarketEvent:
    event_slug: str
    market_slug: str
    sport: str
    title: str
    start_time: Optional[str]
    teams_or_players: List[str]
    probability: float
    last_trade_price: float
    best_bid: float
    best_ask: float
    observed_at: str
    source_confidence: float
    competition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolymarketAdapter:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
        max_events: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or config.POLYMARKET_HTTP_BASE_URL).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else config.BETFAIR_EXTERNAL_SOURCE_HTTP_TIMEOUT_SECONDS
        )
        self.max_events = int(max_events if max_events is not None else config.POLYMARKET_MAX_EVENTS)
        self._headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }

    async def _get_json(self, client: httpx.AsyncClient, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        response = await client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _coerce_probability(value: Any) -> float:
        try:
            val = float(value)
        except Exception:
            return 0.0
        if val > 1.0:
            return max(0.0, min(1.0, val / 100.0))
        return max(0.0, min(1.0, val))

    @staticmethod
    def _extract_participants(title: str) -> List[str]:
        if not title:
            return []
        for token in (" vs ", " v ", " @ ", " at "):
            if token in title.lower():
                parts = [part.strip() for part in title.replace(" VS ", " vs ").split(token) if part.strip()]
                if len(parts) >= 2:
                    return parts[:2]
        return [title]

    @classmethod
    def _parse_event_rows(cls, rows: Iterable[Dict[str, Any]]) -> List[PolymarketEvent]:
        events: List[PolymarketEvent] = []
        observed_at = _utc_now_iso()
        for event_row in rows:
            if not isinstance(event_row, dict):
                continue
            tags = event_row.get("tags") or []
            tag_values = [
                str(tag.get("slug") if isinstance(tag, dict) else tag or "").lower()
                for tag in tags
                if str(tag.get("slug") if isinstance(tag, dict) else tag or "").strip()
            ]
            if config.POLYMARKET_SPORTS_ONLY and "sports" not in tag_values:
                continue
            sport = next((tag for tag in tag_values if tag in _KNOWN_SPORT_TAGS), "unknown")
            event_title = str(event_row.get("title") or event_row.get("slug") or "")
            event_slug = str(event_row.get("slug") or event_title)
            start = _parse_dt(event_row.get("startDate") or event_row.get("start_time"))
            competition = str(event_row.get("seriesSlug") or event_row.get("league") or event_row.get("competition") or "")
            for market_row in event_row.get("markets") or []:
                if not isinstance(market_row, dict):
                    continue
                title = str(market_row.get("question") or event_title or market_row.get("slug") or "")
                market_slug = str(market_row.get("slug") or market_row.get("id") or event_slug)
                last_trade_price = cls._coerce_probability(
                    market_row.get("lastTradePrice") or market_row.get("last_price") or market_row.get("probability")
                )
                best_bid = cls._coerce_probability(market_row.get("bestBid") or market_row.get("best_bid") or last_trade_price)
                best_ask = cls._coerce_probability(market_row.get("bestAsk") or market_row.get("best_ask") or last_trade_price)
                probability = last_trade_price or cls._coerce_probability(market_row.get("price"))
                confidence = 0.7
                if best_bid > 0 and best_ask > 0:
                    spread = abs(best_ask - best_bid)
                    confidence = max(0.3, min(0.95, 0.9 - spread))
                events.append(
                    PolymarketEvent(
                        event_slug=event_slug,
                        market_slug=market_slug,
                        sport=sport or "unknown",
                        title=title,
                        start_time=start.isoformat() if start else None,
                        teams_or_players=cls._extract_participants(title),
                        probability=probability,
                        last_trade_price=last_trade_price,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        observed_at=observed_at,
                        source_confidence=round(confidence, 4),
                        competition=competition,
                    )
                )
        return events

    @classmethod
    def _parse_market_rows(cls, rows: Iterable[Dict[str, Any]]) -> List[PolymarketEvent]:
        synthetic_events: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            synthetic_events.append(
                {
                    "slug": row.get("eventSlug") or row.get("event_slug") or row.get("slug"),
                    "title": row.get("question") or row.get("title") or row.get("slug"),
                    "startDate": row.get("gameStartTime") or row.get("startDate") or row.get("start_time"),
                    "tags": [{"slug": "sports"}, {"slug": row.get("sportsMarketType") or row.get("sport") or "unknown"}],
                    "league": row.get("league") or row.get("competition") or row.get("sportsLeague"),
                    "markets": [dict(row)],
                }
            )
        return cls._parse_event_rows(synthetic_events)

    async def fetch_snapshot(self) -> Dict[str, Any]:
        params = {"closed": "false", "limit": str(self.max_events), "tag_slug": "sports"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=self._headers, follow_redirects=True) as client:
            rows = await self._get_json(client, "/events", params=params)
        if not isinstance(rows, list):
            rows = rows.get("data") or rows.get("events") or []
        events = self._parse_event_rows(rows)
        sports = sorted({event.sport for event in events if event.sport and event.sport != "unknown"})
        return {
            "provider": "polymarket",
            "role": config.POLYMARKET_ROLE,
            "observed_at": _utc_now_iso(),
            "events": [event.to_dict() for event in events],
            "quotes": [event.to_dict() for event in events],
            "sports": sports,
            "event_count": len(events),
            "healthy": bool(events),
            "http_base_url": self.base_url,
            "feed_health": "healthy" if events else "degraded",
        }
