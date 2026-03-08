from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import config
from core.types import PriceSnapshot
from data.candidate_logger import CandidateLogger, build_strategy_record

from betfair.signals.external_event_ingest import ExternalSignalCoordinator
from betfair.signals.external_quote_ingest import build_consensus
from betfair.strategies.crossbook_consensus import evaluate_crossbook_consensus
from betfair.strategies.suspension_lag import evaluate_suspension_lag
from betfair.strategies.timezone_decay import evaluate_timezone_decay


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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


def _candidate_id(strategy_id: str, market_id: str, selection_key: str) -> str:
    raw = f"{strategy_id}|{market_id}|{selection_key}|{int(_utc_now().timestamp())}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cashout_pnl(entry_odds: float, future_lay_odds: float, stake: float = 1.0) -> Optional[float]:
    if entry_odds <= 1.01 or future_lay_odds <= 1.01:
        return None
    hedge_stake = (stake * entry_odds) / future_lay_odds
    pnl = hedge_stake - stake
    return round(pnl, 6)


class BetfairInformationArbManager:
    def __init__(self) -> None:
        self._signals = ExternalSignalCoordinator()
        self._last_refresh_ts: Optional[datetime] = None
        self._refresh_seconds = max(10, int(getattr(config, "BETFAIR_EXTERNAL_REFRESH_SECONDS", 30)))
        self._pending_labels: Dict[str, Dict[str, Any]] = {}
        self._state: Dict[str, Any] = self._empty_state()
        self._seen_keys: Dict[Tuple[str, str, str], datetime] = {}

    @staticmethod
    def _empty_book(strategy_id: str, label: str, explainer: str, mode: str) -> Dict[str, Any]:
        return {
            "strategy_id": strategy_id,
            "label": label,
            "mode": mode,
            "explainer": explainer,
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "realized_net_pnl": 0.0,
            "expected_net_edge": 0.0,
            "fillability_avg": 0.0,
            "top_blockers": [],
            "learning_progress_pct": 0.0,
            "latest_candidates": [],
        }

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "observed_at": None,
            "strategy_books": {
                "betfair_execution_book": self._empty_book(
                    "betfair_execution_book",
                    "Betfair Execution Book",
                    "Executes the currently approved Betfair paper-trading strategies and tracks real paper P&L.",
                    "execute",
                ),
                "betfair_prediction_league": self._empty_book(
                    "betfair_prediction_league",
                    "Prediction League",
                    "Runs prediction models in shadow or canary mode to see which ones deserve execution.",
                    "shadow",
                ),
                "betfair_suspension_lag": self._empty_book(
                    "betfair_suspension_lag",
                    "Suspension-Lag",
                    "Uses fast public event signals and Betfair suspend/resume timing to catch related markets that are still priced for the old game state.",
                    str(getattr(config, "BETFAIR_SUSPENSION_LAG_MODE", "observe")),
                ),
                "betfair_crossbook_consensus": self._empty_book(
                    "betfair_crossbook_consensus",
                    "Cross-Book Consensus",
                    "Compares Betfair prices to a cheap multi-source market consensus and flags places where Betfair looks stale or offside.",
                    str(getattr(config, "BETFAIR_CROSSBOOK_CONSENSUS_MODE", "observe")),
                ),
                "betfair_timezone_decay": self._empty_book(
                    "betfair_timezone_decay",
                    "Timezone Decay",
                    "Targets leagues and time windows where markets are less actively maintained, so stale prices and linked-market mismatches last longer.",
                    str(getattr(config, "BETFAIR_TIMEZONE_DECAY_MODE", "observe")),
                ),
            },
            "polymarket_signal_layer": {
                "label": "Polymarket Sports Signal Layer",
                "explainer": "Uses Polymarket sports market moves as an extra confirmation signal for the same sporting events.",
                "healthy": False,
                "feed_health": "idle",
                "matched_events": 0,
                "unmatched_events": 0,
                "quote_freshness_sec": None,
                "confirmation_hit_rate": 0.0,
                "useful_sports": [],
                "source_health": {},
            },
        }

    async def _refresh_signals(self, market_metadata: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        now = _utc_now()
        if self._last_refresh_ts is None or (now - self._last_refresh_ts).total_seconds() >= self._refresh_seconds:
            self._last_refresh_ts = now
            return await self._signals.refresh(market_metadata)
        return self._signals.state()

    def _book(self, strategy_id: str) -> Dict[str, Any]:
        return self._state["strategy_books"][strategy_id]

    def _register_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        book: Dict[str, Any],
        candidate_logger: Optional[CandidateLogger],
    ) -> None:
        key = (candidate["strategy_id"], candidate["market_id"], candidate["selection_key"])
        last_seen = self._seen_keys.get(key)
        if last_seen is not None and (_utc_now() - last_seen).total_seconds() < 300:
            return
        self._seen_keys[key] = _utc_now()
        candidate = dict(candidate)
        candidate["candidate_id"] = _candidate_id(candidate["strategy_id"], candidate["market_id"], candidate["selection_key"])
        candidate["observed_at"] = _utc_now_iso()
        book["candidate_count"] += 1
        book["accepted_count"] += 1
        book["expected_net_edge"] = round(float(book["expected_net_edge"]) + float(candidate.get("expected_edge", 0.0) or 0.0), 4)
        count = max(1, book["accepted_count"])
        book["fillability_avg"] = round(
            ((float(book["fillability_avg"]) * (count - 1)) + float(candidate.get("fillability_score", 0.0) or 0.0)) / count,
            4,
        )
        book["latest_candidates"] = ([candidate] + book["latest_candidates"])[:8]
        if candidate_logger is not None:
            candidate_logger.log(
                build_strategy_record(
                    strategy_id=candidate["strategy_id"],
                    record_stage="candidate",
                    candidate_id=candidate["candidate_id"],
                    event_name=candidate["event_name"],
                    event_key=candidate["event_key"],
                    market_id=candidate["market_id"],
                    selection_key=candidate["selection_key"],
                    reason=candidate["reason"],
                    confidence=candidate["event_confirmation_level"],
                    signal_strength=float(candidate.get("signal_strength", 0.0) or 0.0),
                    expected_edge=float(candidate.get("expected_edge", 0.0) or 0.0),
                    fillability_score=float(candidate.get("fillability_score", 0.0) or 0.0),
                    source_mix=list(candidate.get("source_mix") or []),
                    external_source_count=int(candidate.get("external_source_count", 0) or 0),
                    polymarket_confirmed=bool(candidate.get("polymarket_confirmed", False)),
                    match_confidence=float(candidate.get("match_confidence", 0.0) or 0.0),
                    quote_freshness_sec=candidate.get("quote_freshness_sec"),
                    event_confirmation_level=candidate["event_confirmation_level"],
                    expected_half_life_sec=candidate.get("expected_half_life_sec"),
                    strategy_context=dict(candidate.get("strategy_context") or {}),
                )
            )
        self._pending_labels[candidate["candidate_id"]] = {
            "candidate": candidate,
            "created_at": _utc_now(),
            "horizons_done": set(),
        }

    def _label_pending(self, snapshots: Dict[str, PriceSnapshot], candidate_logger: Optional[CandidateLogger]) -> None:
        horizon_targets = [int(token.strip()) for token in str(config.BETFAIR_STRATEGY_LOG_LABEL_DELAY_SECONDS).split(",") if token.strip()]
        now = _utc_now()
        to_remove: List[str] = []
        for candidate_id, payload in list(self._pending_labels.items()):
            candidate = payload["candidate"]
            snapshot = snapshots.get(candidate["market_id"])
            if snapshot is None:
                continue
            selection = next((sel for sel in snapshot.selections if str(sel.selection_id) == str(candidate["selection_key"])), None)
            if selection is None:
                continue
            entry_odds = float(candidate.get("entry_back_odds", 0.0) or 0.0)
            future_lay = float(selection.best_lay_price or 0.0) or float(selection.best_back_price or 0.0)
            elapsed = (now - payload["created_at"]).total_seconds()
            updates: Dict[str, Any] = {}
            for horizon in horizon_targets:
                field = f"forward_pnl_{horizon}s"
                if elapsed >= horizon and horizon not in payload["horizons_done"]:
                    updates[field] = _cashout_pnl(entry_odds, future_lay)
                    payload["horizons_done"].add(horizon)
            if not updates:
                continue
            completed = all(horizon in payload["horizons_done"] for horizon in horizon_targets)
            net_pnl = updates.get("forward_pnl_60s")
            if completed:
                book = self._book(candidate["strategy_id"])
                if net_pnl is not None:
                    book["realized_net_pnl"] = round(float(book["realized_net_pnl"]) + float(net_pnl or 0.0), 4)
                to_remove.append(candidate_id)
            if candidate_logger is not None:
                candidate_logger.log(
                    build_strategy_record(
                        strategy_id=candidate["strategy_id"],
                        record_stage="labeled",
                        candidate_id=candidate_id,
                        event_name=candidate["event_name"],
                        event_key=candidate["event_key"],
                        market_id=candidate["market_id"],
                        selection_key=candidate["selection_key"],
                        reason=candidate["reason"],
                        confidence=candidate["event_confirmation_level"],
                        signal_strength=float(candidate.get("signal_strength", 0.0) or 0.0),
                        expected_edge=float(candidate.get("expected_edge", 0.0) or 0.0),
                        fillability_score=float(candidate.get("fillability_score", 0.0) or 0.0),
                        source_mix=list(candidate.get("source_mix") or []),
                        external_source_count=int(candidate.get("external_source_count", 0) or 0),
                        polymarket_confirmed=bool(candidate.get("polymarket_confirmed", False)),
                        match_confidence=float(candidate.get("match_confidence", 0.0) or 0.0),
                        quote_freshness_sec=candidate.get("quote_freshness_sec"),
                        event_confirmation_level=candidate["event_confirmation_level"],
                        expected_half_life_sec=candidate.get("expected_half_life_sec"),
                        forward_pnl_2s=updates.get("forward_pnl_2s"),
                        forward_pnl_5s=updates.get("forward_pnl_5s"),
                        forward_pnl_15s=updates.get("forward_pnl_15s"),
                        forward_pnl_60s=updates.get("forward_pnl_60s"),
                        net_pnl_after_friction=net_pnl,
                        strategy_context=dict(candidate.get("strategy_context") or {}),
                    )
                )
        for candidate_id in to_remove:
            self._pending_labels.pop(candidate_id, None)

    def _progress(self, book: Dict[str, Any], min_target: float = 25.0) -> float:
        candidates = float(book["candidate_count"] or 0)
        return round(min(100.0, (candidates / max(1.0, min_target)) * 100.0), 2)

    async def evaluate_cycle(
        self,
        *,
        market_ids: List[str],
        market_metadata: Dict[str, Dict[str, str]],
        price_cache: Any,
        candidate_logger: Optional[CandidateLogger] = None,
    ) -> Dict[str, Any]:
        if not self._state or not self._state.get("strategy_books"):
            self._state = self._empty_state()
        self._state["observed_at"] = _utc_now_iso()
        signal_state = await self._refresh_signals(market_metadata)
        quotes = list(signal_state.get("quotes") or [])
        matches = list(signal_state.get("matches") or [])
        matches_by_market: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for match in matches:
            for market_id in match.get("betfair_market_ids") or []:
                matches_by_market[str(market_id)].append(match)

        snapshots: Dict[str, PriceSnapshot] = {}
        stale_ratio = 0.0
        for market_id in market_ids:
            snapshot = price_cache.get_prices(market_id)
            if snapshot is None:
                continue
            snapshots[market_id] = snapshot
        active_count = max(1, len(market_ids))
        stale_ratio = round(max(0, active_count - len(snapshots)) / active_count, 4)

        consensus = signal_state.get("consensus") or build_consensus(quotes)
        unmatched = int(signal_state.get("polymarket", {}).get("event_count", 0) or 0) - len(matches)
        self._state["polymarket_signal_layer"].update(
            {
                "healthy": bool(signal_state.get("polymarket", {}).get("healthy", False)),
                "feed_health": "healthy" if signal_state.get("polymarket", {}).get("healthy", False) else "degraded",
                "matched_events": len(matches),
                "unmatched_events": max(0, unmatched),
                "quote_freshness_sec": 0.0 if signal_state.get("polymarket", {}).get("healthy", False) else None,
                "confirmation_hit_rate": round(len(matches) / max(1, int(signal_state.get("polymarket", {}).get("event_count", 0) or 1)), 4),
                "useful_sports": list(signal_state.get("polymarket", {}).get("sports") or []),
                "source_health": signal_state.get("source_health") or {},
                "observed_at": signal_state.get("observed_at"),
            }
        )

        for market_id, snapshot in snapshots.items():
            meta = market_metadata.get(market_id, {})
            for match in matches_by_market.get(market_id, []):
                quote = next(
                    (
                        row for row in quotes
                        if str(row.get("event_slug") or row.get("event_key") or "") == str(match.get("external_event_key") or "")
                    ),
                    {},
                )
                matched_event = dict(match)
                matched_event.update(quote)
                matched_event["quote_freshness_sec"] = 0.0
                candidate = evaluate_suspension_lag(
                    matched_event=matched_event,
                    snapshot=snapshot,
                    market_meta=meta,
                )
                if candidate:
                    self._register_candidate(candidate=candidate, book=self._book("betfair_suspension_lag"), candidate_logger=candidate_logger)
            for consensus_row in consensus.values():
                if str(consensus_row.get("event_key") or "") not in {
                    str(match.get("external_event_key") or "") for match in matches_by_market.get(market_id, [])
                }:
                    continue
                matched_event = matches_by_market.get(market_id, [{}])[0] if matches_by_market.get(market_id) else {}
                candidate = evaluate_crossbook_consensus(
                    consensus_row=consensus_row,
                    snapshot=snapshot,
                    market_meta=meta,
                    matched_event=matched_event,
                )
                if candidate:
                    self._register_candidate(candidate=candidate, book=self._book("betfair_crossbook_consensus"), candidate_logger=candidate_logger)
            candidate = evaluate_timezone_decay(snapshot=snapshot, market_meta=meta, stale_snapshot_ratio=stale_ratio)
            if candidate:
                self._register_candidate(candidate=candidate, book=self._book("betfair_timezone_decay"), candidate_logger=candidate_logger)

        self._label_pending(snapshots, candidate_logger)
        for strategy_id in (
            "betfair_suspension_lag",
            "betfair_crossbook_consensus",
            "betfair_timezone_decay",
        ):
            self._book(strategy_id)["learning_progress_pct"] = self._progress(self._book(strategy_id))
            if strategy_id == "betfair_crossbook_consensus" and not self._book(strategy_id)["candidate_count"]:
                self._book(strategy_id)["top_blockers"] = ["not_enough_external_quote_sources"]
            elif strategy_id == "betfair_suspension_lag" and not self._book(strategy_id)["candidate_count"]:
                self._book(strategy_id)["top_blockers"] = ["awaiting_matched_polymarket_sports_events"]
            elif strategy_id == "betfair_timezone_decay" and not self._book(strategy_id)["candidate_count"]:
                self._book(strategy_id)["top_blockers"] = ["market_ops_signal_below_threshold"]
        return self.state()

    def state(self) -> Dict[str, Any]:
        return dict(self._state)
