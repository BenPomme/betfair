from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import config

from betfair.signals.event_linker import EventLinker
from betfair.signals.external_quote_ingest import build_consensus
from betfair.signals.polymarket_adapter import PolymarketAdapter
from betfair.signals.source_health import SourceHealthTracker
from betfair.signals.thesportsdb_adapter import TheSportsDBAdapter


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExternalEvent:
    source: str
    sport: str
    event_key: str
    event_type: str
    title: str
    participants: List[str]
    scheduled_start: Optional[str]
    observed_at: str
    confidence: float
    raw_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalQuote:
    source: str
    sport: str
    event_key: str
    market_type: str
    selection_key: str
    price: float
    bid: float
    ask: float
    observed_at: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExternalSignalCoordinator:
    def __init__(self) -> None:
        self._health = SourceHealthTracker()
        self._polymarket = PolymarketAdapter()
        self._thesportsdb = TheSportsDBAdapter()
        self._linker = EventLinker()
        self._last_snapshot: Dict[str, Any] = {
            "observed_at": None,
            "events": [],
            "quotes": [],
            "research_events": [],
            "research_quotes": [],
            "matches": [],
            "consensus": {},
            "source_health": {},
            "polymarket": {"healthy": False, "event_count": 0},
            "thesportsdb": {"healthy": False, "event_count": 0},
        }

    @staticmethod
    def _coerce_event_rows(rows: Iterable[Dict[str, Any]], *, source: str) -> List[ExternalEvent]:
        events: List[ExternalEvent] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("market_slug") or row.get("event_slug") or "")
            event_key = str(row.get("event_slug") or row.get("event_key") or title)
            events.append(
                ExternalEvent(
                    source=source,
                    sport=str(row.get("sport") or "unknown"),
                    event_key=str(row.get("event_key") or row.get("event_slug") or title),
                    event_type=str(row.get("event_type") or "price_move_confirmation"),
                    title=title,
                    participants=list(row.get("participants") or row.get("teams_or_players") or []),
                    scheduled_start=row.get("scheduled_start") or row.get("start_time"),
                    observed_at=str(row.get("observed_at") or _utc_now_iso()),
                    confidence=float(row.get("confidence", row.get("source_confidence", 0.5)) or 0.5),
                    raw_payload=dict(row),
                )
            )
        return events

    @staticmethod
    def _coerce_quote_rows(rows: Iterable[Dict[str, Any]]) -> List[ExternalQuote]:
        quotes: List[ExternalQuote] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            event_key = str(row.get("event_slug") or row.get("event_key") or row.get("title") or "")
            selection_key = str(row.get("market_slug") or row.get("selection_key") or row.get("title") or "")
            quotes.append(
                ExternalQuote(
                    source="polymarket",
                    sport=str(row.get("sport") or "unknown"),
                    event_key=event_key,
                    market_type="moneyline",
                    selection_key=selection_key,
                    price=float(row.get("last_trade_price", row.get("probability", 0.0)) or 0.0),
                    bid=float(row.get("best_bid", 0.0) or 0.0),
                    ask=float(row.get("best_ask", 0.0) or 0.0),
                    observed_at=str(row.get("observed_at") or _utc_now_iso()),
                    confidence=float(row.get("source_confidence", 0.5) or 0.5),
                )
            )
        return quotes

    async def refresh(self, market_metadata: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        if not getattr(config, "BETFAIR_EXTERNAL_SIGNALS_ENABLED", True):
            self._last_snapshot = {
                "observed_at": _utc_now_iso(),
                "events": [],
                "quotes": [],
                "research_events": [],
                "research_quotes": [],
                "matches": [],
                "consensus": {},
                "source_health": {},
                "polymarket": {"healthy": False, "disabled": True, "event_count": 0},
                "thesportsdb": {"healthy": False, "disabled": True, "event_count": 0},
            }
            return self._last_snapshot

        events: List[ExternalEvent] = []
        quotes: List[ExternalQuote] = []
        research_events: List[Dict[str, Any]] = []
        research_quotes: List[Dict[str, Any]] = []
        matches: List[Dict[str, Any]] = []
        polymarket_state: Dict[str, Any] = {"healthy": False, "event_count": 0}
        thesportsdb_state: Dict[str, Any] = {"healthy": False, "event_count": 0}
        allowed_sports = {
            str(meta.get("sport_name") or meta.get("sport") or "").strip().lower()
            for meta in (market_metadata or {}).values()
            if str(meta.get("sport_name") or meta.get("sport") or "").strip()
        }
        betfair_events = self._linker.build_betfair_events(market_metadata)

        if getattr(config, "POLYMARKET_ENABLED", True):
            try:
                polymarket_snapshot = await self._polymarket.fetch_snapshot()
                raw_event_rows = list(polymarket_snapshot.get("events") or [])
                raw_quote_rows = list(polymarket_snapshot.get("quotes") or [])
                research_events.extend(raw_event_rows)
                research_quotes.extend(raw_quote_rows)
                if allowed_sports:
                    raw_event_rows = [row for row in raw_event_rows if str(row.get("sport") or "").lower() in allowed_sports]
                    raw_quote_rows = [row for row in raw_quote_rows if str(row.get("sport") or "").lower() in allowed_sports]
                polymarket_state = dict(polymarket_snapshot)
                polymarket_state["filtered_event_count"] = len(raw_event_rows)
                polymarket_state["filtered_quote_count"] = len(raw_quote_rows)
                normalized_events = self._coerce_event_rows(raw_event_rows, source="polymarket")
                events.extend(normalized_events)
                quotes = self._coerce_quote_rows(raw_quote_rows)
                self._health.mark_success("polymarket", item_count=len(normalized_events))
                matches.extend(
                    [
                    match.to_dict()
                    for match in self._linker.match_events(
                        source="polymarket",
                        external_events=[event.to_dict() for event in normalized_events],
                        betfair_events=betfair_events,
                    )
                    ]
                )
            except Exception as exc:
                self._health.mark_error("polymarket", exc)
                polymarket_state = {"healthy": False, "error": str(exc), "event_count": 0}

        if getattr(config, "THESPORTSDB_ENABLED", True):
            try:
                thesportsdb_snapshot = await self._thesportsdb.fetch_snapshot(sorted(allowed_sports))
                raw_event_rows = list(thesportsdb_snapshot.get("events") or [])
                if allowed_sports:
                    raw_event_rows = [row for row in raw_event_rows if str(row.get("sport") or "").lower() in allowed_sports]
                thesportsdb_state = dict(thesportsdb_snapshot)
                thesportsdb_state["filtered_event_count"] = len(raw_event_rows)
                normalized_rows = self._coerce_event_rows(raw_event_rows, source="thesportsdb")
                events.extend(normalized_rows)
                self._health.mark_success("thesportsdb", item_count=len(normalized_rows))
                matches.extend(
                    [
                        match.to_dict()
                        for match in self._linker.match_events(
                            source="thesportsdb",
                            external_events=[event.to_dict() for event in normalized_rows],
                            betfair_events=betfair_events,
                        )
                    ]
                )
            except Exception as exc:
                self._health.mark_error("thesportsdb", exc)
                thesportsdb_state = {"healthy": False, "error": str(exc), "event_count": 0}

        consensus = build_consensus([quote.to_dict() for quote in quotes])
        self._last_snapshot = {
            "observed_at": _utc_now_iso(),
            "events": [event.to_dict() for event in events],
            "quotes": [quote.to_dict() for quote in quotes],
            "research_events": research_events,
            "research_quotes": research_quotes,
            "matches": matches,
            "consensus": consensus,
            "source_health": self._health.snapshot(),
            "polymarket": polymarket_state,
            "thesportsdb": thesportsdb_state,
        }
        return self._last_snapshot

    def state(self) -> Dict[str, Any]:
        return dict(self._last_snapshot)
