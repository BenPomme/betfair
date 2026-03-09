from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import config
from core.types import PriceSnapshot
from data.candidate_logger import CandidateLogger, build_strategy_record
from portfolio.state_store import PortfolioStateStore

from betfair.models.polymarket_binary_ranker import PolymarketBinaryRanker
from betfair.signals.external_event_ingest import ExternalSignalCoordinator
from betfair.signals.external_quote_ingest import build_consensus
from betfair.strategies.crossbook_consensus import evaluate_crossbook_consensus
from betfair.strategies.polymarket_binary_research import (
    build_polymarket_binary_candidates,
    summarize_polymarket_binary_research,
)
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
        self._runtime_store = PortfolioStateStore("betfair_core")
        self._ranker = PolymarketBinaryRanker("betfair_core")
        self._state: Dict[str, Any] = self._empty_state()
        self._seen_keys: Dict[Tuple[str, str, str], datetime] = {}
        self._trade_sequence: int = 0

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
            "open_positions": [],
            "closed_trades": [],
            "recent_events": [],
        }

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "observed_at": None,
            "strategy_books": {
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
                "polymarket_binary_research": self._empty_book(
                    "polymarket_binary_research",
                    "Polymarket Binary Research",
                    "Studies binary-contract spread, liquidity, and repricing pressure on Polymarket to rank research candidates before any trading logic is trusted.",
                    "research",
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
                "source_mix_quality": "single_source",
            },
        }

    def _polymarket_label_summary(self) -> Dict[str, Any]:
        labels = self._ranker.load_labels(limit=5000)
        if not labels:
            return {"count": 0, "win_rate": 0.0, "avg_realized_edge": 0.0, "realized_net_pnl": 0.0, "pending_labels": len(self._ranker.load_pending())}
        wins = sum(1 for row in labels if float(row.get("realized_edge", 0.0) or 0.0) > 0)
        total_edge = sum(float(row.get("realized_edge", 0.0) or 0.0) for row in labels)
        return {
            "count": len(labels),
            "win_rate": round(wins / len(labels), 4),
            "avg_realized_edge": round(total_edge / len(labels), 6),
            "realized_net_pnl": round(total_edge, 6),
            "pending_labels": len(self._ranker.load_pending()),
        }

    @staticmethod
    def _confidence_level(source_count: int, polymarket_confirmed: bool) -> str:
        if source_count >= 2 and polymarket_confirmed:
            return "high"
        if polymarket_confirmed or source_count >= 2:
            return "medium"
        return "low"

    async def _refresh_signals(self, market_metadata: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        now = _utc_now()
        if self._last_refresh_ts is None or (now - self._last_refresh_ts).total_seconds() >= self._refresh_seconds:
            self._last_refresh_ts = now
            return await self._signals.refresh(market_metadata)
        return self._signals.state()

    def _book(self, strategy_id: str) -> Dict[str, Any]:
        return self._state["strategy_books"][strategy_id]

    def _push_book_event(self, book: Dict[str, Any], kind: str, data: Dict[str, Any]) -> None:
        book["recent_events"] = ([{"kind": kind, "data": dict(data)}] + list(book.get("recent_events") or []))[:40]

    def _maybe_open_trade(self, book: Dict[str, Any], candidate: Dict[str, Any]) -> None:
        open_positions = list(book.get("open_positions") or [])
        if len(open_positions) >= int(getattr(config, "BETFAIR_INFO_ARB_MAX_OPEN_POSITIONS_PER_STRATEGY", 5)):
            return
        if any(
            str(item.get("market_id") or "") == str(candidate.get("market_id") or "")
            and str(item.get("selection_key") or "") == str(candidate.get("selection_key") or "")
            for item in open_positions
        ):
            return
        if float(candidate.get("fillability_score", 0.0) or 0.0) < 0.08:
            return
        if float(candidate.get("expected_edge", 0.0) or 0.0) <= 0.0:
            return
        self._trade_sequence += 1
        trade = {
            "trade_id": f"{candidate['strategy_id']}-{self._trade_sequence}",
            "candidate_id": candidate["candidate_id"],
            "market_id": candidate["market_id"],
            "selection_key": candidate["selection_key"],
            "selection_name": candidate.get("selection_name"),
            "symbol": candidate.get("selection_name") or candidate.get("selection_key"),
            "side": str(candidate.get("entry_side") or "back").upper(),
            "status": "OPEN",
            "opened_at": candidate["observed_at"],
            "entry_back_odds": float(candidate.get("entry_back_odds", 0.0) or 0.0),
            "entry_lay_odds": float(candidate.get("entry_lay_odds", 0.0) or 0.0),
            "expected_edge": float(candidate.get("expected_edge", 0.0) or 0.0),
            "signal_strength": float(candidate.get("signal_strength", 0.0) or 0.0),
            "event_name": candidate.get("event_name"),
        }
        candidate["trade_id"] = trade["trade_id"]
        book["open_positions"] = ([trade] + open_positions)[:20]
        self._push_book_event(book, "trade_opened", trade)

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
        self._maybe_open_trade(book, candidate)
        self._push_book_event(book, "candidate_registered", candidate)
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
            longest_horizon = max(payload["horizons_done"]) if payload["horizons_done"] else None
            net_pnl = updates.get(f"forward_pnl_{longest_horizon}s") if longest_horizon is not None else None
            if completed:
                book = self._book(candidate["strategy_id"])
                if net_pnl is not None:
                    book["realized_net_pnl"] = round(float(book["realized_net_pnl"]) + float(net_pnl or 0.0), 4)
                trade_id = candidate.get("trade_id")
                if trade_id:
                    closed_trade = None
                    remaining_open = []
                    for trade in list(book.get("open_positions") or []):
                        if str(trade.get("trade_id") or "") == str(trade_id):
                            closed_trade = {
                                **trade,
                                "status": "CLOSED",
                                "closed_at": _utc_now_iso(),
                                "close_reason": f"horizon_{longest_horizon}s" if longest_horizon is not None else "labeled",
                                "realized_pnl": round(float(net_pnl or 0.0), 6),
                                "net_pnl_usd": round(float(net_pnl or 0.0), 6),
                            }
                        else:
                            remaining_open.append(trade)
                    if closed_trade is not None:
                        book["open_positions"] = remaining_open[:20]
                        book["closed_trades"] = ([closed_trade] + list(book.get("closed_trades") or []))[:40]
                        self._push_book_event(book, "trade_closed", closed_trade)
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
        closed_trades = float(len(book.get("closed_trades") or []))
        progress = min(60.0, (candidates / max(1.0, min_target)) * 60.0)
        progress += min(40.0, (closed_trades / 10.0) * 40.0)
        return round(min(100.0, progress), 2)

    def _persist_polymarket_research_state(self, state: Dict[str, Any]) -> None:
        path = self._runtime_store.runtime_dir / "polymarket_binary_research_state.json"
        self._runtime_store.write_json(path, state)

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
        research_quotes = list(signal_state.get("research_quotes") or quotes)
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
        polymarket_event_count = int(
            signal_state.get("polymarket", {}).get("filtered_event_count")
            or signal_state.get("polymarket", {}).get("event_count", 0)
            or 0
        )
        unmatched = polymarket_event_count - len(matches)
        unique_betfair_events = {
            str(match.get("betfair_event_id") or "")
            for match in matches
            if str(match.get("betfair_event_id") or "")
        }
        matched_polymarket_event_keys = {
            str(match.get("external_event_key") or "")
            for match in matches
            if str(match.get("external_source") or "") == "polymarket"
        }
        self._state["polymarket_signal_layer"].update(
            {
                "healthy": bool(signal_state.get("polymarket", {}).get("healthy", False)),
                "feed_health": "healthy" if signal_state.get("polymarket", {}).get("healthy", False) else "degraded",
                "matched_events": len(unique_betfair_events),
                "unmatched_events": max(0, polymarket_event_count - len(matched_polymarket_event_keys)),
                "quote_freshness_sec": 0.0 if signal_state.get("polymarket", {}).get("healthy", False) else None,
                "confirmation_hit_rate": round(len(matched_polymarket_event_keys) / max(1, polymarket_event_count or 1), 4),
                "useful_sports": list(signal_state.get("polymarket", {}).get("sports") or []),
                "source_health": signal_state.get("source_health") or {},
                "source_mix_quality": "multi_source" if any(str(match.get("external_source")) == "thesportsdb" for match in matches) else "single_source",
                "observed_at": signal_state.get("observed_at"),
                "thesportsdb_event_count": int(signal_state.get("thesportsdb", {}).get("filtered_event_count") or signal_state.get("thesportsdb", {}).get("event_count", 0) or 0),
            }
        )
        label_stats = self._ranker.update_labels(
            current_quotes=research_quotes,
            min_elapsed_seconds=int(getattr(config, "POLYMARKET_BINARY_RESEARCH_LABEL_HORIZON_SECONDS", 120)),
        )
        model_state = self._ranker.load_model()
        raw_binary_candidates = build_polymarket_binary_candidates(research_quotes)
        scored_binary_candidates: List[Dict[str, Any]] = []
        score_threshold = float(getattr(config, "POLYMARKET_BINARY_RESEARCH_MIN_SCORE", -0.01))
        for candidate in raw_binary_candidates:
            score = self._ranker.score_candidate(candidate, model=model_state)
            candidate = dict(candidate)
            candidate["strategy_context"] = {
                **dict(candidate.get("strategy_context") or {}),
                "bucket": score["bucket"],
                "bucket_score": score["bucket_score"],
                "bucket_confidence": score["confidence"],
                "bucket_avg_realized_edge": score["bucket_avg_realized_edge"],
            }
            candidate["expected_edge"] = max(float(candidate.get("expected_edge", 0.0) or 0.0), float(score["empirical_expected_edge"]))
            if float(score["bucket_score"]) < score_threshold and int(score["learned_count"]) >= 5:
                continue
            scored_binary_candidates.append(candidate)
        self._ranker.track_candidates(scored_binary_candidates)
        label_summary = self._polymarket_label_summary()
        label_summary["pending_labels"] = label_stats.get("remaining_pending", label_summary.get("pending_labels", 0))
        polymarket_research = summarize_polymarket_binary_research(
            research_quotes,
            sorted(scored_binary_candidates, key=lambda item: float(item.get("expected_edge", 0.0) or 0.0), reverse=True),
            model_state=model_state,
            label_state=label_summary,
        )
        self._persist_polymarket_research_state(polymarket_research)
        self._state["strategy_books"]["polymarket_binary_research"] = polymarket_research

        for market_id, snapshot in snapshots.items():
            meta = market_metadata.get(market_id, {})
            market_matches = list(matches_by_market.get(market_id, []))
            grouped_sources = sorted({str(match.get("external_source") or "unknown") for match in market_matches})
            matched_event_keys = {
                str(match.get("external_event_key") or "")
                for match in market_matches
                if str(match.get("external_event_key") or "")
            }
            crossbook_match_context: Dict[str, Any] = {}
            if market_matches:
                best_polymarket_match = max(
                    (match for match in market_matches if str(match.get("external_source") or "") == "polymarket"),
                    key=lambda match: float(match.get("match_confidence", 0.0) or 0.0),
                    default=None,
                )
                best_market_match = max(
                    market_matches,
                    key=lambda match: float(match.get("match_confidence", 0.0) or 0.0),
                )
                crossbook_match_context = dict(best_market_match)
                crossbook_match_context["source_mix"] = grouped_sources
                crossbook_match_context["external_source_count"] = len(grouped_sources)
                crossbook_match_context["polymarket_confirmed"] = "polymarket" in grouped_sources
                crossbook_match_context["event_confirmation_level"] = self._confidence_level(
                    len(grouped_sources),
                    bool(crossbook_match_context["polymarket_confirmed"]),
                )
                if best_polymarket_match is not None:
                    quote = next(
                        (
                            row for row in quotes
                            if str(row.get("event_slug") or row.get("event_key") or "") == str(best_polymarket_match.get("external_event_key") or "")
                        ),
                        {},
                    )
                    matched_event = dict(best_polymarket_match)
                    matched_event.update(quote)
                    matched_event["quote_freshness_sec"] = 0.0
                    matched_event["source_mix"] = ["betfair_suspend_resume"] + grouped_sources
                    matched_event["external_source_count"] = len(grouped_sources)
                    matched_event["polymarket_confirmed"] = "polymarket" in grouped_sources
                    matched_event["event_confirmation_level"] = self._confidence_level(
                        len(grouped_sources),
                        bool(matched_event["polymarket_confirmed"]),
                    )
                    candidate = evaluate_suspension_lag(
                        matched_event=matched_event,
                        snapshot=snapshot,
                        market_meta=meta,
                    )
                    if candidate:
                        self._register_candidate(candidate=candidate, book=self._book("betfair_suspension_lag"), candidate_logger=candidate_logger)
            for consensus_row in consensus.values():
                if str(consensus_row.get("event_key") or "") not in matched_event_keys:
                    continue
                candidate = evaluate_crossbook_consensus(
                    consensus_row=consensus_row,
                    snapshot=snapshot,
                    market_meta=meta,
                    matched_event=crossbook_match_context,
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
                self._book(strategy_id)["top_blockers"] = ["awaiting_multi_source_event_confirmation"]
            elif strategy_id == "betfair_timezone_decay" and not self._book(strategy_id)["candidate_count"]:
                self._book(strategy_id)["top_blockers"] = ["market_ops_signal_below_threshold"]
        return self.state()

    def state(self) -> Dict[str, Any]:
        return dict(self._state)
