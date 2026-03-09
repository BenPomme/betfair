from __future__ import annotations

from datetime import timedelta
import re
from typing import Any, Dict, Iterable, List, Optional
import logging

import httpx

import config
from polymarket.utils import clamp, parse_ts, safe_json_list, to_float, utc_now, utc_now_iso

logger = logging.getLogger(__name__)
_SPORTS_TAG_ID = "1"
_PARTICIPANT_SPLITTERS = (
    re.compile(r"\s+vs\.?\s+", re.IGNORECASE),
    re.compile(r"\s+versus\s+", re.IGNORECASE),
    re.compile(r"\s+@\s+", re.IGNORECASE),
)

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
    "college-basketball",
    "ncaab",
    "cfb",
    "ncaaf",
}


def _tag_values(row: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for tag in row.get("tags") or []:
        if isinstance(tag, dict):
            slug = str(tag.get("slug") or "").strip().lower()
        else:
            slug = str(tag or "").strip().lower()
        if slug:
            values.append(slug)
    return values


def _participants(title: str) -> List[str]:
    text = " ".join(str(title or "").strip().split())
    if not text:
        return []
    for splitter in _PARTICIPANT_SPLITTERS:
        parts = [part.strip(" -") for part in splitter.split(text) if part.strip(" -")]
        if len(parts) >= 2 and all(len(part) >= 2 for part in parts[:2]):
            return parts[:2]
    return [text]


class PolymarketGammaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
        max_events: Optional[int] = None,
        sports_filter: Optional[Iterable[str]] = None,
    ) -> None:
        self.base_url = (base_url or config.POLYMARKET_HTTP_BASE_URL).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else config.BETFAIR_EXTERNAL_SOURCE_HTTP_TIMEOUT_SECONDS
        )
        self.max_events = int(max_events if max_events is not None else getattr(config, "POLYMARKET_QF_MAX_EVENTS", config.POLYMARKET_MAX_EVENTS))
        self.sports_filter = {
            str(item).strip().lower()
            for item in (sports_filter or str(getattr(config, "POLYMARKET_QF_SPORTS_FILTER", "soccer,basketball,tennis,football,baseball,hockey")).split(","))
            if str(item).strip()
        }
        self._headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }

    @staticmethod
    def normalize_events(rows: Iterable[Dict[str, Any]], sports_filter: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        observed_at = utc_now_iso()
        events: List[Dict[str, Any]] = []
        markets: List[Dict[str, Any]] = []
        tokens: List[Dict[str, Any]] = []
        sport_filter_set = {
            str(item).strip().lower()
            for item in (sports_filter or [])
            if str(item).strip()
        }
        for event_row in rows:
            if not isinstance(event_row, dict):
                continue
            tags = _tag_values(event_row)
            if getattr(config, "POLYMARKET_SPORTS_ONLY", True) and "sports" not in tags:
                continue
            sport = next(
                (
                    str(value).lower()
                    for value in (
                        event_row.get("sport"),
                        event_row.get("sportsMarketType"),
                        *tags,
                    )
                    if str(value).lower() in _KNOWN_SPORT_TAGS
                ),
                "unknown",
            )
            if sport_filter_set and sport not in sport_filter_set:
                continue
            event_title = str(event_row.get("title") or event_row.get("slug") or "")
            event_slug = str(event_row.get("slug") or event_row.get("id") or event_title)
            start_time = parse_ts(event_row.get("startDate") or event_row.get("start_time") or event_row.get("gameStartTime"))
            event_end_time = parse_ts(event_row.get("endDate") or event_row.get("end_time"))
            competition = str(
                event_row.get("seriesSlug")
                or event_row.get("league")
                or event_row.get("competition")
                or event_row.get("sportsLeague")
                or ""
            )
            event_payload = {
                "event_id": str(event_row.get("id") or event_slug),
                "event_slug": event_slug,
                "title": event_title,
                "sport": sport,
                "competition": competition,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": event_end_time.isoformat() if event_end_time else None,
                "teams_or_players": _participants(event_title),
                "observed_at": observed_at,
                "source_confidence": 0.8,
            }
            events.append(event_payload)
            for market_row in event_row.get("markets") or []:
                if not isinstance(market_row, dict):
                    continue
                market_slug = str(market_row.get("slug") or market_row.get("id") or event_slug)
                question = str(market_row.get("question") or market_row.get("title") or event_title or market_slug)
                token_ids = [str(item) for item in safe_json_list(market_row.get("clobTokenIds")) if str(item).strip()]
                outcomes = [str(item) for item in safe_json_list(market_row.get("outcomes")) if str(item).strip()]
                outcome_prices = [clamp(to_float(item)) for item in safe_json_list(market_row.get("outcomePrices"))]
                if len(token_ids) != 2:
                    continue
                if len(outcomes) != len(token_ids):
                    outcomes = [f"outcome_{idx}" for idx in range(len(token_ids))]
                if len(outcome_prices) != len(token_ids):
                    last_trade = clamp(to_float(market_row.get("lastTradePrice") or market_row.get("last_trade_price")))
                    if len(token_ids) == 2 and last_trade > 0:
                        outcome_prices = [last_trade, clamp(1.0 - last_trade)]
                    else:
                        outcome_prices = [0.0 for _ in token_ids]
                market_end_time = parse_ts(
                    market_row.get("endDate") or event_row.get("endDate") or market_row.get("end_time")
                )
                market_payload = {
                    "event_slug": event_slug,
                    "market_id": str(market_row.get("id") or market_slug),
                    "market_slug": market_slug,
                    "question": question,
                    "sport": sport,
                    "competition": competition,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": market_end_time.isoformat() if market_end_time else None,
                    "observed_at": observed_at,
                    "liquidity": to_float(market_row.get("liquidityNum") or market_row.get("liquidity")),
                    "volume_24hr": to_float(market_row.get("volume24hrClob") or market_row.get("volume24hr") or market_row.get("volume")),
                    "volume_1wk": to_float(market_row.get("volume1wkClob") or market_row.get("volume1wk")),
                    "closed": bool(market_row.get("closed", False)),
                    "resolved": bool(market_row.get("resolved", False)),
                    "resolution": market_row.get("resolution"),
                    "token_ids": token_ids,
                    "outcomes": outcomes,
                }
                markets.append(market_payload)
                for index, token_id in enumerate(token_ids):
                    gamma_price = clamp(outcome_prices[index] if index < len(outcome_prices) else 0.0)
                    token_payload = {
                        "event_id": event_payload["event_id"],
                        "event_slug": event_slug,
                        "market_id": market_payload["market_id"],
                        "market_slug": market_slug,
                        "selection_key": f"{market_slug}:{index}",
                        "token_id": token_id,
                        "token_index": index,
                        "outcome": outcomes[index] if index < len(outcomes) else f"outcome_{index}",
                        "title": question,
                        "sport": sport,
                        "competition": competition,
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": market_payload["end_time"],
                        "teams_or_players": event_payload["teams_or_players"],
                        "gamma_price": gamma_price,
                        "last_trade_price": gamma_price,
                        "best_bid": clamp(to_float(market_row.get("bestBid") or 0.0)),
                        "best_ask": clamp(to_float(market_row.get("bestAsk") or 0.0)),
                        "liquidity": market_payload["liquidity"],
                        "volume_24hr": market_payload["volume_24hr"],
                        "volume_1wk": market_payload["volume_1wk"],
                        "resolved": market_payload["resolved"],
                        "closed": market_payload["closed"],
                        "resolution": market_payload["resolution"],
                        "observed_at": observed_at,
                    }
                    tokens.append(token_payload)
        sports = sorted({str(item.get("sport")) for item in events if item.get("sport") and item.get("sport") != "unknown"})
        return {
            "provider": "polymarket_gamma",
            "observed_at": observed_at,
            "healthy": bool(tokens),
            "events": events,
            "markets": markets,
            "tokens": tokens,
            "sports": sports,
            "event_count": len(events),
            "market_count": len(markets),
            "token_count": len(tokens),
        }

    def fetch_snapshot(self) -> Dict[str, Any]:
        page_size = max(1, min(self.max_events, 100))
        now = utc_now()
        end_date_max = now + timedelta(hours=max(1, int(getattr(config, "POLYMARKET_QF_MAX_SETTLEMENT_HOURS", 168))))
        params = {
            "active": "true",
            "closed": "false",
            "limit": str(page_size),
            "tag_id": _SPORTS_TAG_ID,
            "end_date_min": now.isoformat().replace("+00:00", "Z"),
            "end_date_max": end_date_max.isoformat().replace("+00:00", "Z"),
        }
        rows: List[Dict[str, Any]] = []
        with httpx.Client(timeout=self.timeout_seconds, headers=self._headers, follow_redirects=True) as client:
            offset = 0
            while len(rows) < self.max_events:
                response = client.get(f"{self.base_url}/events", params={**params, "offset": str(offset)})
                response.raise_for_status()
                payload = response.json()
                batch = payload if isinstance(payload, list) else (payload.get("data") or payload.get("events") or [])
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
        rows = rows[: self.max_events]
        snapshot = self.normalize_events(rows, sports_filter=self.sports_filter)
        snapshot["source_health"] = {
            "healthy": snapshot["healthy"],
            "base_url": self.base_url,
            "event_count": snapshot["event_count"],
            "market_count": snapshot["market_count"],
            "token_count": snapshot["token_count"],
            "discovery_window_hours": int(getattr(config, "POLYMARKET_QF_MAX_SETTLEMENT_HOURS", 168)),
        }
        return snapshot
