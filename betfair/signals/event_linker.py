from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import config


_STOPWORDS = {
    "fc",
    "cf",
    "club",
    "team",
    "match",
    "winner",
    "game",
    "the",
    "vs",
    "v",
}


def _normalize_token(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in _STOPWORDS]
    return " ".join(tokens)


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _time_score(left: Optional[datetime], right: Optional[datetime]) -> float:
    if left is None or right is None:
        return 0.5
    delta_minutes = abs((left - right).total_seconds()) / 60.0
    if delta_minutes <= 5:
        return 1.0
    if delta_minutes <= 30:
        return 0.8
    if delta_minutes <= 120:
        return 0.5
    return 0.0


def _name_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


@dataclass(frozen=True)
class EventMatch:
    external_source: str
    external_event_key: str
    betfair_event_id: str
    betfair_market_ids: List[str]
    match_confidence: float
    match_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventLinker:
    def __init__(self, min_confidence: Optional[float] = None):
        self.min_confidence = float(
            min_confidence if min_confidence is not None else getattr(config, "BETFAIR_EVENT_MATCH_MIN_CONFIDENCE", 0.8)
        )

    @staticmethod
    def build_betfair_events(market_metadata: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        events: Dict[str, Dict[str, Any]] = {}
        for market_id, meta in (market_metadata or {}).items():
            event_id = str(meta.get("event_id") or meta.get("event_name") or market_id)
            row = events.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "event_name": str(meta.get("event_name") or ""),
                    "sport": str(meta.get("sport_name") or meta.get("sport") or ""),
                    "competition": str(meta.get("competition_name") or ""),
                    "market_start": meta.get("market_start"),
                    "market_ids": [],
                },
            )
            row["market_ids"].append(market_id)
            if not row.get("market_start") and meta.get("market_start"):
                row["market_start"] = meta.get("market_start")
        return events

    def match_events(
        self,
        *,
        source: str,
        external_events: Iterable[Dict[str, Any]],
        betfair_events: Dict[str, Dict[str, Any]],
    ) -> List[EventMatch]:
        matches: List[EventMatch] = []
        for event in external_events:
            event_key = str(event.get("event_key") or event.get("title") or "")
            ext_name = _normalize_token(str(event.get("title") or ""))
            ext_sport = _normalize_token(str(event.get("sport") or ""))
            ext_start = _parse_utc(event.get("scheduled_start"))
            best: Optional[EventMatch] = None
            for betfair_event_id, betfair in betfair_events.items():
                bf_name = _normalize_token(str(betfair.get("event_name") or ""))
                bf_sport = _normalize_token(str(betfair.get("sport") or ""))
                name_score = _name_score(ext_name, bf_name)
                sport_score = 1.0 if ext_sport and bf_sport and ext_sport == bf_sport else 0.5
                time_score = _time_score(ext_start, _parse_utc(betfair.get("market_start")))
                competition_name = _normalize_token(str(betfair.get("competition") or ""))
                league_score = _name_score(_normalize_token(str(event.get("competition") or "")), competition_name) if competition_name else 0.5
                confidence = round((name_score * 0.55) + (sport_score * 0.15) + (time_score * 0.2) + (league_score * 0.1), 4)
                if best is None or confidence > best.match_confidence:
                    reason = f"name={name_score:.2f},sport={sport_score:.2f},time={time_score:.2f},league={league_score:.2f}"
                    best = EventMatch(
                        external_source=source,
                        external_event_key=event_key,
                        betfair_event_id=betfair_event_id,
                        betfair_market_ids=list(betfair.get("market_ids") or []),
                        match_confidence=confidence,
                        match_reason=reason,
                    )
            if best and best.match_confidence >= self.min_confidence:
                matches.append(best)
        return matches
